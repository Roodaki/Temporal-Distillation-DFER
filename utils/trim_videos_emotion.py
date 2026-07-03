import cv2
import pandas as pd
import os
import pathlib
import time
import multiprocessing as mp

source_videos_directory = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\2. Video Facial Emotion Recognition (VFER)\Dataset\DFEW_face"
pre_analyzed_data_directory = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\2. Video Facial Emotion Recognition (VFER)\Dataset\DFEW_face_analyze"
trimmed_output_directory = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\2. Video Facial Emotion Recognition (VFER)\Dataset\DFEW_face_trimmed32"

CLASS_TO_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}

VIDEO_EXTENSIONS = [".mp4"]
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

TRIM_WINDOW_SIZE = 32
NUM_TOP_SEGMENTS_TO_SELECT = 3
MINIMUM_SCORE_FOR_TOP_N_THRESHOLD = 1.0

FOURCC_CODEC = cv2.VideoWriter_fourcc(*"mp4v")
OUTPUT_VIDEO_EXTENSION = ".mp4"


def find_analysis_folders_imagenet(
    pre_analyzed_root: pathlib.Path,
) -> list[pathlib.Path]:
    """Find every *_analyze directory anywhere under an ImageNet-style analysis root."""
    return sorted(p for p in pre_analyzed_root.rglob("*_analyze") if p.is_dir())


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


# --- Worker Function for processing a single video's analysis ---
def process_single_video_worker(task_args_tuple):
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
    saved_segments_for_this_video = 0

    try:
        relative_class_dir = analysis_folder_path.parent.relative_to(pre_analyzed_root)
    except ValueError:
        return f"{video_stem}: Error - Analysis folder is not under the analysis root."

    # In ImageNet-style datasets, the class comes from the folder name, not from the filename.
    # If there are nested folders, the immediate parent is treated as the class label.
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

    cap_props = cv2.VideoCapture(str(original_video_file))
    if not cap_props.isOpened():
        return f"{relative_class_dir}/{video_stem}: Error - Could not open original video for properties."

    fps = cap_props.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap_props.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_props.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_original_video = int(cap_props.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_props.release()

    if fps <= 0 or frame_width <= 0 or frame_height <= 0:
        return f"{relative_class_dir}/{video_stem}: Error - Invalid video properties."

    can_trim_based_on_data = len(df) >= trim_window_size
    can_trim_based_on_video = total_frames_original_video >= trim_window_size
    if not (can_trim_based_on_data and can_trim_based_on_video):
        return (
            f"{relative_class_dir}/{video_stem}: Skipping - Not enough frames in CSV "
            f"({len(df)}) or video ({total_frames_original_video}) for window {trim_window_size}."
        )

    all_evaluated_segments = []
    max_start_index_csv = len(df) - trim_window_size
    max_start_index_video = total_frames_original_video - trim_window_size

    for i in range(min(max_start_index_csv, max_start_index_video) + 1):
        window_df = df.iloc[i : i + trim_window_size]
        window_mean_target_emotion = window_df[target_emotion].mean()
        if pd.isna(window_mean_target_emotion):
            continue
        all_evaluated_segments.append(
            {
                "df_start_index": i,
                "mean_emotion_score": window_mean_target_emotion,
                "video_start_frame": i + 1,
                "video_end_frame": i + trim_window_size,
            }
        )

    if not all_evaluated_segments:
        return (
            f"{relative_class_dir}/{video_stem}: No valid segments could be evaluated."
        )

    all_evaluated_segments.sort(key=lambda x: x["mean_emotion_score"], reverse=True)
    qualified_segments = [
        segment
        for segment in all_evaluated_segments[:num_top_segments]
        if segment["mean_emotion_score"] >= min_score_threshold
    ]

    if not qualified_segments:
        return (
            f"{relative_class_dir}/{video_stem}: No segments from top {num_top_segments} "
            f"met score threshold {min_score_threshold}."
        )

    filename_suffix_counter = 0
    for segment_info in qualified_segments:
        filename_suffix_counter += 1
        trim_start_frame_number = segment_info["video_start_frame"]

        trimmed_video_name = (
            f"{video_stem}_trimmed_{filename_suffix_counter}{output_video_extension}"
        )
        trimmed_video_path = class_output_dir / trimmed_video_name

        out_writer = cv2.VideoWriter(
            str(trimmed_video_path), fourcc_codec, fps, (frame_width, frame_height)
        )
        if not out_writer.isOpened():
            print(
                f"[{pid}-{relative_class_dir}/{video_stem}] Error: VideoWriter failed for {trimmed_video_name}. Skipping."
            )
            filename_suffix_counter -= 1
            continue

        cap_trim = cv2.VideoCapture(str(original_video_file))
        segment_ok = True

        if not cap_trim.isOpened():
            print(
                f"[{pid}-{relative_class_dir}/{video_stem}] Error: Could not re-open {original_video_file}."
            )
            segment_ok = False
        else:
            # OpenCV frame index is zero-based; video_start_frame above is one-based.
            cap_trim.set(cv2.CAP_PROP_POS_FRAMES, trim_start_frame_number - 1)

            for _ in range(trim_window_size):
                ret_read, frame_to_write = cap_trim.read()
                if not ret_read:
                    segment_ok = False
                    break
                out_writer.write(frame_to_write)

            cap_trim.release()

        out_writer.release()

        if segment_ok:
            saved_segments_for_this_video += 1
        else:
            filename_suffix_counter -= 1
            if trimmed_video_path.exists():
                try:
                    os.remove(trimmed_video_path)
                except Exception as e_rem:
                    print(
                        f"[{pid}-{relative_class_dir}/{video_stem}] Error removing incomplete file {trimmed_video_path}: {e_rem}"
                    )

    processing_time = time.time() - start_time_video_processing
    summary = (
        f"{relative_class_dir}/{video_stem}: Processed in {processing_time:.2f}s. "
        f"Saved {saved_segments_for_this_video} segments."
    )
    print(f"[{pid}-{relative_class_dir}/{video_stem}] Finished. {summary}")
    return summary


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Start method already set or cannot be forced. Continuing...")

    start_time_script = time.time()

    source_videos_root = pathlib.Path(source_videos_directory)
    pre_analyzed_root = pathlib.Path(pre_analyzed_data_directory)
    trimmed_output_root = pathlib.Path(trimmed_output_directory)

    if not source_videos_root.is_dir():
        print(f"Error: Source videos directory not found: {source_videos_root}")
        raise SystemExit(1)

    if not pre_analyzed_root.is_dir():
        print(f"Error: Pre-analyzed data directory not found: {pre_analyzed_root}")
        raise SystemExit(1)

    trimmed_output_root.mkdir(parents=True, exist_ok=True)
    print(f"Emotion-guided trimmed videos will be saved in: {trimmed_output_root}")

    analysis_folders_found = find_analysis_folders_imagenet(pre_analyzed_root)
    if not analysis_folders_found:
        print(f"No '*_analyze' folders found under {pre_analyzed_root}.")
        raise SystemExit(1)

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
            TRIM_WINDOW_SIZE,
            NUM_TOP_SEGMENTS_TO_SELECT,
            MINIMUM_SCORE_FOR_TOP_N_THRESHOLD,
            FOURCC_CODEC,
            OUTPUT_VIDEO_EXTENSION,
        )
        for folder_path in analysis_folders_found
    ]

    num_processes_to_use = max(1, mp.cpu_count() - 1)
    print(f"Starting parallel processing with {num_processes_to_use} processes...")

    with mp.Pool(processes=num_processes_to_use) as pool:
        results = pool.map(process_single_video_worker, tasks_args_list)

    print("\n" + "=" * 50)
    print("Parallel Processing Summary:")

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
    print(f"All processing finished in {time.time() - start_time_script:.2f} seconds.")
    print(f"Trimmed videos are in: {trimmed_output_root}")
