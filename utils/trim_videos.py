import cv2
import pandas as pd
import os
import pathlib
import time
import numpy as np
import multiprocessing as mp  # Import multiprocessing

# --- Configuration --- (Remains the same)
source_videos_directory = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Dataset\rotated_face224"
pre_analyzed_data_directory = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Dataset\rotated_face224_analyze"
trimmed_output_directory = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Dataset\rotated_face224_trimmed32"  # New output path

CLASS_TO_EMOTION_MAP = {
    "amusement": "happy",
    "anger": "angry",
    "awe": "surprise",
    "disgust": "disgust",
    "enthusiasm": "happy",
    "fear": "fear",
    "liking": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}
video_extensions_list_const = [
    ".mp4"
]  # Renamed to avoid conflict if 'video_extensions' is used as var name
emotions_list_const = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

TRIM_WINDOW_SIZE_const = 32
NUM_TOP_SEGMENTS_TO_SELECT_const = 40
MINIMUM_SCORE_FOR_TOP_N_THRESHOLD_const = 1.0  # (0-100 scale)

FOURCC_CODEC_const = cv2.VideoWriter_fourcc(*"mp4v")
OUTPUT_VIDEO_EXTENSION_const = ".mp4"
# --- End of Configuration ---


# --- Worker Function for processing a single video's analysis ---
def process_single_video_worker(task_args_tuple):
    # Unpack arguments
    (
        analysis_folder_path,
        source_videos_path_str,
        trimmed_output_path_str,  # Pass paths as strings or re-Pathify
        class_to_emotion_map_dict,
        video_extensions_list_arg,
        emotions_list_arg,
        trim_window_size_arg,
        num_top_segments_arg,
        min_score_threshold_arg,
        fourcc_codec_arg,
        output_video_extension_arg,
    ) = task_args_tuple

    # Re-create Path objects if strings were passed, or ensure they are path-like
    source_videos_path = pathlib.Path(source_videos_path_str)
    trimmed_output_path = pathlib.Path(trimmed_output_path_str)

    video_stem = analysis_folder_path.name.replace("_analyze", "")
    start_time_video_processing = time.time()
    pid = os.getpid()

    # Initialize a counter for successfully saved segments for THIS video
    actual_segments_successfully_saved_for_this_video = 0

    # --- Start of logic adapted from the original loop for a single video ---
    csv_file_path = analysis_folder_path / "emotion_data.csv"
    if not csv_file_path.exists():
        # print(f"[{pid}-{video_stem}] Error: 'emotion_data.csv' not found. Skipping.")
        return f"{video_stem}: Error - CSV not found."
    try:
        df = pd.read_csv(csv_file_path)
        if df.empty or "frame" not in df.columns:
            # print(f"[{pid}-{video_stem}] Warning: CSV is empty or missing 'frame' column. Skipping.")
            return f"{video_stem}: Error - CSV empty or bad format."
    except Exception as e:
        # print(f"[{pid}-{video_stem}] Error reading CSV {csv_file_path}: {e}. Skipping.")
        return f"{video_stem}: Error - CSV read failed: {e}"

    original_video_file = None
    for ext in video_extensions_list_arg:
        potential_video_path = source_videos_path / (video_stem + ext)
        if potential_video_path.exists():
            original_video_file = potential_video_path
            break
    if not original_video_file:
        # print(f"[{pid}-{video_stem}] Error: Original video not found. Skipping.")
        return f"{video_stem}: Error - Original video not found."

    parts = video_stem.split("_")
    target_emotion_column = None
    if len(parts) > 1:
        video_class = parts[1].lower()
        if video_class in class_to_emotion_map_dict:
            target_emotion = class_to_emotion_map_dict[video_class]
            if target_emotion in emotions_list_arg and target_emotion in df.columns:
                target_emotion_column = target_emotion
            else:  # Mapped emotion not in CSV or predefined list
                return f"{video_stem}: Error - Target emotion '{target_emotion}' for class '{video_class}' not valid/found in CSV."
        else:  # Class not in map
            return f"{video_stem}: Error - Video class '{video_class}' not in CLASS_TO_EMOTION_MAP."
    else:  # Filename format issue
        return (
            f"{video_stem}: Error - Video stem format incorrect for class extraction."
        )
    if (
        not target_emotion_column
    ):  # Should be caught by returns above, but as a safeguard
        return f"{video_stem}: Error - Could not determine target emotion column."

    cap_props = cv2.VideoCapture(str(original_video_file))
    if not cap_props.isOpened():
        return f"{video_stem}: Error - Could not open original video for properties."
    fps = cap_props.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap_props.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_props.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_original_video = int(cap_props.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_props.release()

    can_trim_based_on_data = len(df) >= trim_window_size_arg
    can_trim_based_on_video = total_frames_original_video >= trim_window_size_arg
    if not (can_trim_based_on_data and can_trim_based_on_video):
        return f"{video_stem}: Skipping - Not enough frames in CSV ({len(df)}) or video ({total_frames_original_video}) for window {trim_window_size_arg}."

    all_evaluated_segments = []
    max_start_index_csv = len(df) - trim_window_size_arg
    max_start_index_video = total_frames_original_video - trim_window_size_arg
    for i in range(min(max_start_index_csv, max_start_index_video) + 1):
        window_df = df.iloc[i : i + trim_window_size_arg]
        window_mean_target_emotion = window_df[target_emotion_column].mean()
        if pd.isna(window_mean_target_emotion):
            continue
        all_evaluated_segments.append(
            {
                "df_start_index": i,
                "mean_emotion_score": window_mean_target_emotion,
                "video_start_frame": i + 1,
                "video_end_frame": i + 1 + trim_window_size_arg - 1,
            }
        )

    if not all_evaluated_segments:
        return f"{video_stem}: No valid segments could be evaluated."

    all_evaluated_segments.sort(key=lambda x: x["mean_emotion_score"], reverse=True)
    top_n_segments_before_threshold = all_evaluated_segments[:num_top_segments_arg]

    qualified_segments_to_save = []
    for (
        segment_info_loop
    ) in top_n_segments_before_threshold:  # Renamed to avoid conflict
        if segment_info_loop["mean_emotion_score"] >= min_score_threshold_arg:
            qualified_segments_to_save.append(segment_info_loop)

    if not qualified_segments_to_save:
        return f"{video_stem}: No segments from top {num_top_segments_arg} met score threshold {min_score_threshold_arg}."

    # This counter is for the filename suffix of successfully written files meeting all criteria.
    filename_suffix_counter = 0
    for segment_info in qualified_segments_to_save:  # Iterate through the final list
        filename_suffix_counter += (
            1  # This will be 1, 2, 3... for segments being attempted
        )

        trim_start_frame_number = segment_info["video_start_frame"]
        # mean_score_for_segment = segment_info['mean_emotion_score'] # For printing if needed

        # Simplified print for parallel, more detail can be added if desired
        # print(f"[{pid}-{video_stem}] Attempting save #{filename_suffix_counter} (Score: {mean_score_for_segment:.2f}) Frames: {segment_info['video_start_frame']}-{segment_info['video_end_frame']}")

        trimmed_video_name = f"{video_stem}_trimmed_{filename_suffix_counter}{output_video_extension_arg}"
        trimmed_video_path = trimmed_output_path / trimmed_video_name

        out_writer = cv2.VideoWriter(
            str(trimmed_video_path), fourcc_codec_arg, fps, (frame_width, frame_height)
        )
        if not out_writer.isOpened():
            print(
                f"[{pid}-{video_stem}] Error: VideoWriter failed for {trimmed_video_name}. Skipping."
            )
            filename_suffix_counter -= 1  # This segment won't be saved with this number
            continue

        cap_trim = cv2.VideoCapture(str(original_video_file))
        segment_trim_successful_flag = True  # Renamed for clarity
        if not cap_trim.isOpened():
            print(
                f"[{pid}-{video_stem}] Error: Could not re-open {original_video_file} for {trimmed_video_name}. Skipping."
            )
            out_writer.release()
            segment_trim_successful_flag = False
            filename_suffix_counter -= 1
        else:
            for _ in range(trim_start_frame_number - 1):  # Seek
                ret_seek, _ = cap_trim.read()
                if not ret_seek:
                    segment_trim_successful_flag = False
                    break

            if segment_trim_successful_flag:
                frames_written = 0
                for _ in range(trim_window_size_arg):
                    ret_read, frame_to_write = cap_trim.read()
                    if not ret_read:
                        segment_trim_successful_flag = False
                        break
                    out_writer.write(frame_to_write)
                    frames_written += 1

                if segment_trim_successful_flag:
                    # print(f"[{pid}-{video_stem}] Saved {trimmed_video_name} ({frames_written} frames).")
                    actual_segments_successfully_saved_for_this_video += (
                        1  # Increment the video's success counter
                    )
                # else: segment failed during write
            # else: segment failed during seek
            cap_trim.release()
        out_writer.release()

        if not segment_trim_successful_flag:
            filename_suffix_counter -= (
                1  # Correct suffix for next attempt if current one failed completely
            )
            if os.path.exists(trimmed_video_path):
                # print(f"[{pid}-{video_stem}] Removing incomplete: {trimmed_video_path}")
                try:
                    os.remove(trimmed_video_path)
                except Exception as e_rem:
                    print(
                        f"[{pid}-{video_stem}] Error removing {trimmed_video_path}: {e_rem}"
                    )

    # --- End of logic adapted from the original loop ---
    processing_time = time.time() - start_time_video_processing
    summary = f"{video_stem}: Processed in {processing_time:.2f}s. Saved {actual_segments_successfully_saved_for_this_video} segments."
    print(f"[{pid}-{video_stem}] Finished. {summary}")
    return summary


# --- End of Worker Function ---


if __name__ == "__main__":
    # Ensure this is called only once at the beginning if using "spawn" or "forkserver"
    # On Windows, "spawn" is often default. On Linux/macOS, "fork" is default.
    # For cross-platform consistency and to avoid issues with some libraries, "spawn" can be good.
    try:
        mp.set_start_method("spawn", force=True)  # Call this early
    except RuntimeError:
        print("Start method already set or cannot be forced. Continuing...")
        pass

    start_time_script = time.time()

    source_videos_path = pathlib.Path(source_videos_directory)
    if not source_videos_path.is_dir():
        print(f"Error: Source videos directory not found: {source_videos_path}")
        exit()

    pre_analyzed_path = pathlib.Path(pre_analyzed_data_directory)
    if not pre_analyzed_path.is_dir():
        print(f"Error: Pre-analyzed data directory not found: {pre_analyzed_path}")
        exit()

    trimmed_output_path = pathlib.Path(trimmed_output_directory)
    trimmed_output_path.mkdir(parents=True, exist_ok=True)
    print(
        f"Top-N (then thresholded, parallel) trimmed videos will be saved in: {trimmed_output_path}"
    )

    analysis_folders_found = [
        p for p in pre_analyzed_path.glob("*_analyze") if p.is_dir()
    ]
    if not analysis_folders_found:
        print(f"No '_analyze' folders found in {pre_analyzed_path}.")
        exit()
    print(f"Found {len(analysis_folders_found)} pre-analyzed folders to process.")

    # Prepare arguments for each task
    # Pass paths as strings because pathlib objects might not always pickle perfectly across all OS/mp start methods
    # Although they often do, strings are safer for arguments.
    tasks_args_list = []
    for folder_path in analysis_folders_found:
        tasks_args_list.append(
            (
                folder_path,
                str(source_videos_path),
                str(trimmed_output_path),
                CLASS_TO_EMOTION_MAP,
                video_extensions_list_const,
                emotions_list_const,
                TRIM_WINDOW_SIZE_const,
                NUM_TOP_SEGMENTS_TO_SELECT_const,
                MINIMUM_SCORE_FOR_TOP_N_THRESHOLD_const,
                FOURCC_CODEC_const,
                OUTPUT_VIDEO_EXTENSION_const,
            )
        )

    if not tasks_args_list:
        print("No tasks to process.")
        exit()

    num_processes_to_use = (
        mp.cpu_count()
    )  # Or set a fixed number e.g., max(1, mp.cpu_count() -1)
    print(f"Starting parallel processing with {num_processes_to_use} processes...")

    with mp.Pool(processes=num_processes_to_use) as pool:
        # Using starmap as process_single_video_worker takes a tuple which needs to be unpacked
        # If process_single_video_worker took a single list/tuple arg, map would be fine
        # The worker function takes a single argument 'task_args_tuple', so 'map' is correct.
        # Each element in tasks_args_list IS that single tuple argument.
        results = pool.map(process_single_video_worker, tasks_args_list)

    print("\n" + "=" * 50)
    print("Parallel Processing Summary:")
    total_videos_processed_successfully = 0
    total_segments_saved_overall = 0

    for i, result_summary in enumerate(results):
        print(f"  Result for task {i+1}: {result_summary}")
        if "Error" not in result_summary:  # Basic success check
            total_videos_processed_successfully += 1
            try:  # Try to parse segments saved from summary string
                # Example: "video_stem: Processed in X.Xs. Saved Y segments."
                segments_part = result_summary.split("Saved ")[1]
                num_saved = int(segments_part.split(" ")[0])
                total_segments_saved_overall += num_saved
            except (IndexError, ValueError) as e:
                # print(f"    Could not parse segment count from: {result_summary} due to {e}")
                pass  # If parsing fails, just don't add to total_segments_saved_overall

    print(
        f"\nSuccessfully processed (no fatal errors reported by worker) for {total_videos_processed_successfully} out of {len(tasks_args_list)} videos."
    )
    print(
        f"Total segments successfully written across all videos: {total_segments_saved_overall}"
    )
    print(
        f"All pre-analyzed data processed in {time.time() - start_time_script:.2f} seconds."
    )
    print(f"Trimmed videos are in: {trimmed_output_path}")
