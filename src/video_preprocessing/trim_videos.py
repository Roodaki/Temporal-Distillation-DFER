import argparse
import cv2
import pandas as pd
import os
import pathlib
import time
import numpy as np
import multiprocessing as mp

# =========================================================================
# Shared configuration
# =========================================================================

CLASS_TO_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

NUM_TOP_SEGMENTS_TO_SELECT = 10
MINIMUM_SCORE_FOR_TOP_N_THRESHOLD = 1.0

NEUTRAL_EMOTION_COLUMN = "neutral"
MAXIMUM_SCORE_FOR_MIN_NEUTRAL_THRESHOLD = 0.9

NUM_RANDOM_SEGMENTS_TO_SAMPLE = NUM_TOP_SEGMENTS_TO_SELECT

FOURCC_CODEC = cv2.VideoWriter_fourcc(*"mp4v")
OUTPUT_VIDEO_EXTENSION = ".mp4"


# =========================================================================
# Shared helpers
# =========================================================================


def find_analysis_folders_imagenet(
    pre_analyzed_root: pathlib.Path,
) -> list[pathlib.Path]:
    """Find every *_analyze directory anywhere under an ImageNet-style analysis root."""
    return sorted(p for p in pre_analyzed_root.rglob("*_analyze") if p.is_dir())


def collect_videos_imagenet(
    source_root: pathlib.Path, video_extensions: list[str]
) -> list[pathlib.Path]:
    """Collect videos from class subfolders under an ImageNet-style dataset root."""
    suffixes = {ext.lower() for ext in video_extensions}
    return sorted(
        p
        for p in source_root.rglob("*")
        if p.is_file() and p.suffix.lower() in suffixes
    )


def find_source_video(
    source_root: pathlib.Path,
    relative_class_dir: pathlib.Path,
    video_stem: str,
    video_extensions: list[str],
) -> pathlib.Path | None:
    """Find the original video under source_root / relative_class_dir."""
    class_source_dir = source_root / relative_class_dir
    for ext in video_extensions:
        candidate = class_source_dir / f"{video_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_saved_segment_count(result_summary: str) -> int:
    try:
        return int(result_summary.split("Saved ")[1].split(" ")[0])
    except (IndexError, ValueError):
        return 0


def read_video_properties(video_path: pathlib.Path):
    """Open a video just to read fps/width/height/frame-count, then release it."""
    cap_props = cv2.VideoCapture(str(video_path))
    if not cap_props.isOpened():
        return None
    fps = cap_props.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap_props.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_props.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_props.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_props.release()
    if fps <= 0 or frame_width <= 0 or frame_height <= 0:
        return None
    return fps, frame_width, frame_height, total_frames


def write_segment(
    original_video_file: pathlib.Path,
    output_path: pathlib.Path,
    start_frame_zero_based: int,
    trim_window_size: int,
    fps: float,
    frame_width: int,
    frame_height: int,
    fourcc_codec,
    log_prefix: str,
) -> bool:
    """Read `trim_window_size` frames starting at start_frame_zero_based and write them out.
    Returns True on success, False otherwise (and cleans up any partial file)."""
    out_writer = cv2.VideoWriter(
        str(output_path), fourcc_codec, fps, (frame_width, frame_height)
    )
    if not out_writer.isOpened():
        print(
            f"[{log_prefix}] Error: VideoWriter failed for {output_path.name}. Skipping."
        )
        return False

    cap = cv2.VideoCapture(str(original_video_file))
    segment_ok = True

    if not cap.isOpened():
        print(f"[{log_prefix}] Error: Could not open {original_video_file}.")
        segment_ok = False
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_zero_based)
        for _ in range(trim_window_size):
            ret, frame = cap.read()
            if not ret:
                segment_ok = False
                break
            out_writer.write(frame)
        cap.release()

    out_writer.release()

    if not segment_ok and output_path.exists():
        try:
            os.remove(output_path)
        except Exception as e_rem:
            print(
                f"[{log_prefix}] Error removing incomplete file {output_path}: {e_rem}"
            )

    return segment_ok


def evaluate_sliding_windows(
    df: pd.DataFrame,
    score_column: str,
    trim_window_size: int,
    total_frames_original_video: int,
) -> list[dict]:
    """Compute the mean of `score_column` over every valid sliding window and
    return a list of {df_start_index, mean_score, video_start_frame, video_end_frame}.
    """
    max_start_index_csv = len(df) - trim_window_size
    max_start_index_video = total_frames_original_video - trim_window_size

    rolling_means = df[score_column].rolling(window=trim_window_size).mean()

    evaluated_segments = []
    last_valid_start = min(max_start_index_csv, max_start_index_video)
    for i in range(last_valid_start + 1):
        # rolling mean at position (i + trim_window_size - 1) covers window [i, i+trim_window_size)
        window_mean_score = rolling_means.iloc[i + trim_window_size - 1]
        if pd.isna(window_mean_score):
            continue
        evaluated_segments.append(
            {
                "df_start_index": i,
                "mean_score": window_mean_score,
                "video_start_frame": i + 1,
                "video_end_frame": i + trim_window_size,
            }
        )
    return evaluated_segments


def select_non_overlapping_segments(
    ordered_segments: list[dict],
    num_segments_to_select: int,
) -> list[dict]:
    """Greedy temporal non-max suppression.

    `ordered_segments` must already be sorted by priority (best candidate
    first — e.g. highest/lowest mean score, or a random shuffle). Walk the
    list in that order; keep a candidate only if it shares no frame with any
    segment already selected, i.e. reject it if
    candidate.start <= chosen.end AND candidate.end >= chosen.start.
    Stop once `num_segments_to_select` non-overlapping segments are picked
    or the candidate list is exhausted.
    """
    selected_segments: list[dict] = []
    for candidate in ordered_segments:
        if len(selected_segments) >= num_segments_to_select:
            break
        overlaps_existing = any(
            candidate["video_start_frame"] <= chosen["video_end_frame"]
            and candidate["video_end_frame"] >= chosen["video_start_frame"]
            for chosen in selected_segments
        )
        if not overlaps_existing:
            selected_segments.append(candidate)
    return selected_segments


def save_qualified_segments(
    qualified_segments: list[dict],
    original_video_file: pathlib.Path,
    class_output_dir: pathlib.Path,
    video_stem: str,
    filename_suffix: str,
    trim_window_size: int,
    fps: float,
    frame_width: int,
    frame_height: int,
    fourcc_codec,
    output_video_extension: str,
    log_prefix: str,
) -> int:
    """Write out each qualified segment, returning the number successfully saved."""
    saved_count = 0
    filename_suffix_counter = 0
    for segment_info in qualified_segments:
        filename_suffix_counter += 1
        # video_start_frame is one-based; writer needs zero-based.
        start_frame_zero_based = segment_info["video_start_frame"] - 1

        trimmed_video_name = f"{video_stem}_{filename_suffix}_{filename_suffix_counter}{output_video_extension}"
        trimmed_video_path = class_output_dir / trimmed_video_name

        ok = write_segment(
            original_video_file,
            trimmed_video_path,
            start_frame_zero_based,
            trim_window_size,
            fps,
            frame_width,
            frame_height,
            fourcc_codec,
            log_prefix,
        )

        if ok:
            saved_count += 1
        else:
            filename_suffix_counter -= 1

    return saved_count


# =========================================================================
# Max-emotion worker (formerly "emotion" mode)
# =========================================================================


def process_single_video_max_emotion_worker(task_args_tuple):
    (
        analysis_folder_path_str,
        pre_analyzed_root_str,
        source_videos_root_str,
        trimmed_output_root_str,
        class_to_emotion_map,
        video_extensions,
        emotions,
        trim_window_size,
        num_top_segments,
        min_score_threshold,
        fourcc_codec,
        output_video_extension,
    ) = task_args_tuple

    analysis_folder_path = pathlib.Path(analysis_folder_path_str)
    pre_analyzed_root = pathlib.Path(pre_analyzed_root_str)
    source_videos_root = pathlib.Path(source_videos_root_str)
    trimmed_output_root = pathlib.Path(trimmed_output_root_str)

    video_stem = analysis_folder_path.name.replace("_analyze", "")
    pid = os.getpid()
    start_time_video_processing = time.time()

    try:
        relative_class_dir = analysis_folder_path.parent.relative_to(pre_analyzed_root)
    except ValueError:
        return f"{video_stem}: Error - Analysis folder is not under the analysis root."

    log_prefix = f"{pid}-{relative_class_dir}/{video_stem}"

    # In ImageNet-style datasets, the class comes from the folder name, not from the filename.
    video_class = analysis_folder_path.parent.name.lower()
    if video_class not in class_to_emotion_map:
        return f"{relative_class_dir}/{video_stem}: Error - Class folder '{video_class}' not in CLASS_TO_EMOTION_MAP."

    target_emotion = class_to_emotion_map[video_class]
    if target_emotion not in emotions:
        return f"{relative_class_dir}/{video_stem}: Error - Mapped emotion '{target_emotion}' is not valid."

    csv_file_path = analysis_folder_path / "emotion_data.csv"
    if not csv_file_path.exists():
        return f"{relative_class_dir}/{video_stem}: Error - CSV not found."

    try:
        df = pd.read_csv(csv_file_path)
        if df.empty or "frame" not in df.columns:
            return f"{relative_class_dir}/{video_stem}: Error - CSV empty or missing 'frame' column."
        if target_emotion not in df.columns:
            return f"{relative_class_dir}/{video_stem}: Error - Target emotion column '{target_emotion}' not found in CSV."
    except Exception as e:
        return f"{relative_class_dir}/{video_stem}: Error - CSV read failed: {e}"

    original_video_file = find_source_video(
        source_videos_root, relative_class_dir, video_stem, video_extensions
    )
    if original_video_file is None:
        return f"{relative_class_dir}/{video_stem}: Error - Original video not found under source class folder."

    class_output_dir = trimmed_output_root / relative_class_dir
    class_output_dir.mkdir(parents=True, exist_ok=True)

    props = read_video_properties(original_video_file)
    if props is None:
        return f"{relative_class_dir}/{video_stem}: Error - Could not open/read original video properties."
    fps, frame_width, frame_height, total_frames_original_video = props

    can_trim_based_on_data = len(df) >= trim_window_size
    can_trim_based_on_video = total_frames_original_video >= trim_window_size
    if not (can_trim_based_on_data and can_trim_based_on_video):
        return (
            f"{relative_class_dir}/{video_stem}: Skipping - Not enough frames in CSV "
            f"({len(df)}) or video ({total_frames_original_video}) for window {trim_window_size}."
        )

    all_evaluated_segments = evaluate_sliding_windows(
        df, target_emotion, trim_window_size, total_frames_original_video
    )

    if not all_evaluated_segments:
        return (
            f"{relative_class_dir}/{video_stem}: No valid segments could be evaluated."
        )

    # Highest mean target-emotion score first.
    all_evaluated_segments.sort(key=lambda x: x["mean_score"], reverse=True)
    non_overlapping_candidates = select_non_overlapping_segments(
        all_evaluated_segments, num_top_segments
    )
    qualified_segments = [
        segment
        for segment in non_overlapping_candidates
        if segment["mean_score"] >= min_score_threshold
    ]

    if not qualified_segments:
        return (
            f"{relative_class_dir}/{video_stem}: No segments from top {num_top_segments} "
            f"met score threshold {min_score_threshold}."
        )

    saved_segments_for_this_video = save_qualified_segments(
        qualified_segments,
        original_video_file,
        class_output_dir,
        video_stem,
        "max_emotion",
        trim_window_size,
        fps,
        frame_width,
        frame_height,
        fourcc_codec,
        output_video_extension,
        log_prefix,
    )

    processing_time = time.time() - start_time_video_processing
    summary = (
        f"{relative_class_dir}/{video_stem}: Processed in {processing_time:.2f}s. "
        f"Saved {saved_segments_for_this_video} segments."
    )
    print(f"[{log_prefix}] Finished. {summary}")
    return summary


# =========================================================================
# Min-neutral worker
# =========================================================================


def process_single_video_min_neutral_worker(task_args_tuple):
    (
        analysis_folder_path_str,
        pre_analyzed_root_str,
        source_videos_root_str,
        trimmed_output_root_str,
        video_extensions,
        neutral_column,
        trim_window_size,
        num_top_segments,
        max_score_threshold,
        fourcc_codec,
        output_video_extension,
    ) = task_args_tuple

    analysis_folder_path = pathlib.Path(analysis_folder_path_str)
    pre_analyzed_root = pathlib.Path(pre_analyzed_root_str)
    source_videos_root = pathlib.Path(source_videos_root_str)
    trimmed_output_root = pathlib.Path(trimmed_output_root_str)

    video_stem = analysis_folder_path.name.replace("_analyze", "")
    pid = os.getpid()
    start_time_video_processing = time.time()

    try:
        relative_class_dir = analysis_folder_path.parent.relative_to(pre_analyzed_root)
    except ValueError:
        return f"{video_stem}: Error - Analysis folder is not under the analysis root."

    log_prefix = f"{pid}-{relative_class_dir}/{video_stem}"

    csv_file_path = analysis_folder_path / "emotion_data.csv"
    if not csv_file_path.exists():
        return f"{relative_class_dir}/{video_stem}: Error - CSV not found."

    try:
        df = pd.read_csv(csv_file_path)
        if df.empty or "frame" not in df.columns:
            return f"{relative_class_dir}/{video_stem}: Error - CSV empty or missing 'frame' column."
        if neutral_column not in df.columns:
            return f"{relative_class_dir}/{video_stem}: Error - '{neutral_column}' column not found in CSV."
    except Exception as e:
        return f"{relative_class_dir}/{video_stem}: Error - CSV read failed: {e}"

    original_video_file = find_source_video(
        source_videos_root, relative_class_dir, video_stem, video_extensions
    )
    if original_video_file is None:
        return f"{relative_class_dir}/{video_stem}: Error - Original video not found under source class folder."

    class_output_dir = trimmed_output_root / relative_class_dir
    class_output_dir.mkdir(parents=True, exist_ok=True)

    props = read_video_properties(original_video_file)
    if props is None:
        return f"{relative_class_dir}/{video_stem}: Error - Could not open/read original video properties."
    fps, frame_width, frame_height, total_frames_original_video = props

    can_trim_based_on_data = len(df) >= trim_window_size
    can_trim_based_on_video = total_frames_original_video >= trim_window_size
    if not (can_trim_based_on_data and can_trim_based_on_video):
        return (
            f"{relative_class_dir}/{video_stem}: Skipping - Not enough frames in CSV "
            f"({len(df)}) or video ({total_frames_original_video}) for window {trim_window_size}."
        )

    all_evaluated_segments = evaluate_sliding_windows(
        df, neutral_column, trim_window_size, total_frames_original_video
    )

    if not all_evaluated_segments:
        return (
            f"{relative_class_dir}/{video_stem}: No valid segments could be evaluated."
        )

    # Lowest mean neutral score first (least neutral / most expressive).
    all_evaluated_segments.sort(key=lambda x: x["mean_score"])
    non_overlapping_candidates = select_non_overlapping_segments(
        all_evaluated_segments, num_top_segments
    )
    qualified_segments = [
        segment
        for segment in non_overlapping_candidates
        if segment["mean_score"] <= max_score_threshold
    ]

    if not qualified_segments:
        return (
            f"{relative_class_dir}/{video_stem}: No segments from bottom {num_top_segments} "
            f"met neutral-score ceiling {max_score_threshold}."
        )

    saved_segments_for_this_video = save_qualified_segments(
        qualified_segments,
        original_video_file,
        class_output_dir,
        video_stem,
        "min_neutral",
        trim_window_size,
        fps,
        frame_width,
        frame_height,
        fourcc_codec,
        output_video_extension,
        log_prefix,
    )

    processing_time = time.time() - start_time_video_processing
    summary = (
        f"{relative_class_dir}/{video_stem}: Processed in {processing_time:.2f}s. "
        f"Saved {saved_segments_for_this_video} segments."
    )
    print(f"[{log_prefix}] Finished. {summary}")
    return summary


# =========================================================================
# Random-trimming worker
# =========================================================================


def process_single_video_random_worker(task_args_tuple):
    (
        video_path_str,
        source_root_str,
        trimmed_output_root_str,
        trim_window_size,
        num_segments,
        fourcc_codec,
        output_video_extension,
    ) = task_args_tuple

    video_path = pathlib.Path(video_path_str)
    source_root = pathlib.Path(source_root_str)
    trimmed_output_root = pathlib.Path(trimmed_output_root_str)
    video_stem = video_path.stem
    pid = os.getpid()
    start_time = time.time()

    try:
        relative_class_dir = video_path.parent.relative_to(source_root)
    except ValueError:
        return f"{video_stem}: Error - Video is not under the source root."

    log_prefix = f"{pid}-{relative_class_dir}/{video_stem}"

    class_output_dir = trimmed_output_root / relative_class_dir
    class_output_dir.mkdir(parents=True, exist_ok=True)

    props = read_video_properties(video_path)
    if props is None:
        return f"{relative_class_dir}/{video_stem}: Error - Could not open/read video properties."
    fps, frame_width, frame_height, total_frames = props

    if total_frames < trim_window_size:
        return (
            f"{relative_class_dir}/{video_stem}: Skipped - Video too short "
            f"({total_frames} frames) for window {trim_window_size}."
        )

    # Build every valid window as a candidate (same shape as the score-based
    # workers use), then shuffle so "priority order" is random instead of
    # score order, and run it through the same NMS-style greedy selector so
    # random picks never overlap each other either.
    max_valid_start_index = total_frames - trim_window_size
    all_candidate_segments = [
        {
            "video_start_frame": start_idx + 1,
            "video_end_frame": start_idx + trim_window_size,
        }
        for start_idx in range(max_valid_start_index + 1)
    ]
    np.random.shuffle(all_candidate_segments)

    selected_segments = select_non_overlapping_segments(
        all_candidate_segments, num_segments
    )
    # Keep chronological order in the output filenames/order.
    selected_segments.sort(key=lambda seg: seg["video_start_frame"])

    segments_saved_count = 0
    for i, segment_info in enumerate(selected_segments, start=1):
        start_frame_zero_based = segment_info["video_start_frame"] - 1
        output_name = f"{video_stem}_rnd_{i}{output_video_extension}"
        output_file_path = class_output_dir / output_name

        ok = write_segment(
            video_path,
            output_file_path,
            start_frame_zero_based,
            trim_window_size,
            fps,
            frame_width,
            frame_height,
            fourcc_codec,
            log_prefix,
        )
        if ok:
            segments_saved_count += 1

    duration = time.time() - start_time
    return (
        f"{relative_class_dir}/{video_stem}: Saved {segments_saved_count} random segments "
        f"in {duration:.2f}s."
    )


# =========================================================================
# Pass runners
# =========================================================================


def run_max_emotion_pass(args, trim_window_size):
    source_videos_root = pathlib.Path(args.source_dir)
    pre_analyzed_root = pathlib.Path(args.logs)
    trimmed_output_root = pathlib.Path(args.output_dir) / "max_emotion"

    if not source_videos_root.is_dir():
        print(f"Error: Source videos directory not found: {source_videos_root}")
        return
    if not pre_analyzed_root.is_dir():
        print(f"Error: Pre-analyzed data directory not found: {pre_analyzed_root}")
        return

    trimmed_output_root.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Max-emotion pass ===")
    print(f"Max-emotion trimmed videos will be saved in: {trimmed_output_root}")

    analysis_folders_found = find_analysis_folders_imagenet(pre_analyzed_root)
    if not analysis_folders_found:
        print(f"No '*_analyze' folders found under {pre_analyzed_root}.")
        return

    print(f"Found {len(analysis_folders_found)} pre-analyzed folders to process.")

    tasks_args_list = [
        (
            str(folder_path),
            str(pre_analyzed_root),
            str(source_videos_root),
            str(trimmed_output_root),
            CLASS_TO_EMOTION_MAP,
            VIDEO_EXTENSIONS,
            EMOTIONS,
            trim_window_size,
            NUM_TOP_SEGMENTS_TO_SELECT,
            MINIMUM_SCORE_FOR_TOP_N_THRESHOLD,
            FOURCC_CODEC,
            OUTPUT_VIDEO_EXTENSION,
        )
        for folder_path in analysis_folders_found
    ]

    num_processes_to_use = max(1, mp.cpu_count() - 1)
    print(f"Starting parallel processing with {num_processes_to_use} processes...")

    start_time_script = time.time()
    with mp.Pool(processes=num_processes_to_use) as pool:
        results = pool.map(process_single_video_max_emotion_worker, tasks_args_list)

    print("\n" + "=" * 50)
    print("Max-emotion Pass Summary:")

    successful_results = 0
    total_segments_saved = 0

    for i, result_summary in enumerate(results, start=1):
        print(f"  Result for task {i}: {result_summary}")
        if "Error" not in result_summary and "Skipping" not in result_summary:
            successful_results += 1
            total_segments_saved += parse_saved_segment_count(result_summary)

    print(
        f"\nSuccessfully processed {successful_results} out of {len(tasks_args_list)} analysis folders."
    )
    print(f"Total segments successfully written: {total_segments_saved}")
    print(
        f"Max-emotion pass finished in {time.time() - start_time_script:.2f} seconds."
    )
    print(f"Trimmed videos are in: {trimmed_output_root}")


def run_min_neutral_pass(args, trim_window_size):
    source_videos_root = pathlib.Path(args.source_dir)
    pre_analyzed_root = pathlib.Path(args.logs)
    trimmed_output_root = pathlib.Path(args.output_dir) / "min_neutral"

    if not source_videos_root.is_dir():
        print(f"Error: Source videos directory not found: {source_videos_root}")
        return
    if not pre_analyzed_root.is_dir():
        print(f"Error: Pre-analyzed data directory not found: {pre_analyzed_root}")
        return

    trimmed_output_root.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Min-neutral pass ===")
    print(f"Min-neutral trimmed videos will be saved in: {trimmed_output_root}")

    analysis_folders_found = find_analysis_folders_imagenet(pre_analyzed_root)
    if not analysis_folders_found:
        print(f"No '*_analyze' folders found under {pre_analyzed_root}.")
        return

    print(f"Found {len(analysis_folders_found)} pre-analyzed folders to process.")

    tasks_args_list = [
        (
            str(folder_path),
            str(pre_analyzed_root),
            str(source_videos_root),
            str(trimmed_output_root),
            VIDEO_EXTENSIONS,
            NEUTRAL_EMOTION_COLUMN,
            trim_window_size,
            NUM_TOP_SEGMENTS_TO_SELECT,
            MAXIMUM_SCORE_FOR_MIN_NEUTRAL_THRESHOLD,
            FOURCC_CODEC,
            OUTPUT_VIDEO_EXTENSION,
        )
        for folder_path in analysis_folders_found
    ]

    num_processes_to_use = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    print(f"Starting parallel processing with {num_processes_to_use} processes...")

    start_time_script = time.time()
    with mp.Pool(processes=num_processes_to_use) as pool:
        results = pool.map(process_single_video_min_neutral_worker, tasks_args_list)

    print("\n" + "=" * 50)
    print("Min-neutral Pass Summary:")

    successful_results = 0
    total_segments_saved = 0

    for i, result_summary in enumerate(results, start=1):
        print(f"  Result for task {i}: {result_summary}")
        if "Error" not in result_summary and "Skipping" not in result_summary:
            successful_results += 1
            total_segments_saved += parse_saved_segment_count(result_summary)

    print(
        f"\nSuccessfully processed {successful_results} out of {len(tasks_args_list)} analysis folders."
    )
    print(f"Total segments successfully written: {total_segments_saved}")
    print(
        f"Min-neutral pass finished in {time.time() - start_time_script:.2f} seconds."
    )
    print(f"Trimmed videos are in: {trimmed_output_root}")


def run_random_pass(args, trim_window_size):
    source_root = pathlib.Path(args.source_dir)
    trimmed_output_root = pathlib.Path(args.output_dir) / "random"
    trimmed_output_root.mkdir(parents=True, exist_ok=True)

    if not source_root.is_dir():
        print(f"Error: Source not found: {source_root}")
        return

    print(f"\n=== Random pass ===")
    video_files = collect_videos_imagenet(source_root, VIDEO_EXTENSIONS)

    if not video_files:
        print(f"No video files found under ImageNet-style root: {source_root}")
        return

    print(f"Found {len(video_files)} videos under: {source_root}")
    print(f"Outputting to: {trimmed_output_root}")
    print(
        f"Configuration: {NUM_RANDOM_SEGMENTS_TO_SAMPLE} random segments of size {trim_window_size} per video."
    )

    tasks = [
        (
            str(video_path),
            str(source_root),
            str(trimmed_output_root),
            trim_window_size,
            NUM_RANDOM_SEGMENTS_TO_SAMPLE,
            FOURCC_CODEC,
            OUTPUT_VIDEO_EXTENSION,
        )
        for video_path in video_files
    ]

    num_procs = max(1, mp.cpu_count() - 1)
    print(f"Starting pool with {num_procs} processes...")

    start_script = time.time()
    with mp.Pool(processes=num_procs) as pool:
        results = pool.map(process_single_video_random_worker, tasks)

    success_count = 0
    total_segments = 0

    print("\n" + "=" * 50)
    print("Random Pass Summary:")
    for res in results:
        print(f"  Result: {res}")
        if "Saved" in res:
            success_count += 1
            total_segments += parse_saved_segment_count(res)

    print(f"\nProcessed {success_count}/{len(video_files)} videos successfully.")
    print(f"Total random clips generated: {total_segments}")
    print(f"Random pass finished in {time.time() - start_script:.2f}s")
    print(f"Trimmed videos are in: {trimmed_output_root}")


# =========================================================================
# CLI
# =========================================================================


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal distillation: max-emotion, min-neutral, and/or random trimming."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["max_emotion", "min_neutral", "random", "both"],
        default="both",
        help="Which trimming strategy to run. 'both' runs all three passes "
        "(max_emotion, min_neutral, random).",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="ImageNet-style root of cropped-face videos (input to analyze_videos.py).",
    )
    parser.add_argument(
        "--logs",
        type=str,
        required=True,
        help="Root of per-video emotion CSV logs produced by analyze_videos.py. "
        "Only required for --mode max_emotion, min_neutral, or both.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory. Max-emotion clips go in <output_dir>/max_emotion, "
        "min-neutral clips go in <output_dir>/min_neutral, "
        "random clips go in <output_dir>/random.",
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        required=True,
        help="Number of frames per distilled clip (e.g. 16 for ViViT, 8 for TimeSformer).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Start method already set or cannot be forced. Continuing...")

    args = get_args()
    TRIM_WINDOW_SIZE = args.clip_length

    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()

    if args.mode in ("max_emotion", "both"):
        run_max_emotion_pass(args, TRIM_WINDOW_SIZE)

    if args.mode in ("min_neutral", "both"):
        run_min_neutral_pass(args, TRIM_WINDOW_SIZE)

    if args.mode in ("random", "both"):
        run_random_pass(args, TRIM_WINDOW_SIZE)

    print("\n" + "=" * 50)
    print(
        f"All requested passes finished in {time.time() - overall_start:.2f} seconds."
    )
    print(f"Outputs are under: {output_root}")