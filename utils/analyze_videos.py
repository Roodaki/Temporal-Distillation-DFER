import cv2
from deepface import DeepFace
import pandas as pd
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
import time
import pathlib
import numpy as np

# --- Configuration ---
# Set the path to the folder containing the source video files
source_videos_directory = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Dataset\rotated_face448"  # <--- Change this to the folder path

# Set the parent directory where the [video_name]_analyze folder will be created.
analysis_output_parent_directory = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Dataset\rotated_face448_analyze"


# We only process MP4 files now
video_extensions = [".mp4"]

# Emotions deepface usually detects:
emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# --- Accuracy Enhancement Configuration ---
# Choose a face detector backend. More accurate backends can improve detection.
# Options: 'opencv', 'retinaface', 'mtcnn', 'ssd', 'dlib', 'mediapipe', 'yolov8', 'yunet'.
# 'retinaface', 'mtcnn', 'yolov8', 'yunet', 'mediapipe' are often more accurate.
# Some backends might require additional installations (e.g., 'yolov8' needs 'ultralytics', 'mediapipe' needs 'mediapipe').
# Experiment to find the best one for your dataset.
DETECTOR_BACKEND = "mtcnn"  # <--- Change this to experiment

# Number of processes to use. Defaults to the number of CPU cores.
NUM_PROCESSES = mp.cpu_count()


# --- Worker Function for Parallel Processing ---
def analyze_single_frame(frame_tuple):
    """Analyzes a single frame for emotions."""
    frame_count, frame = frame_tuple
    # Ensure 'emotions' list is accessible if it's not passed or global in worker's context
    # For this script structure, 'emotions' is a global variable accessible by the worker.

    try:
        # Using the configured DETECTOR_BACKEND and ensuring alignment is on (default)
        # enforce_detection=False: if no face, DeepFace won't raise an error,
        # but analysis_results might be empty or lack 'emotion'.
        analysis_results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,  # Keeps current behavior: script continues, records zeros if no face
            align=True,  # Face alignment is generally good for accuracy (default is True)
        )

        if (
            analysis_results
            and isinstance(analysis_results, list)
            and len(analysis_results) > 0
        ):
            face_result = analysis_results[
                0
            ]  # DeepFace returns a list of dicts (taking the first detected face)
            emotions_in_frame = face_result.get("emotion", {})  # Use .get for safety

            frame_entry = {"frame": frame_count}
            for emotion_key in emotions:  # Use the global 'emotions' list
                frame_entry[emotion_key] = emotions_in_frame.get(emotion_key, 0.0)
            return (frame_count, frame_entry)
        else:  # No face detected or empty results
            frame_entry = {"frame": frame_count}
            for emotion_key in emotions:
                frame_entry[emotion_key] = 0.0
            return (frame_count, frame_entry)

    except Exception as e:
        # It's good to know which frame in which process failed.
        print(
            f"Error processing frame {frame_count} in worker (PID {os.getpid()}) with backend {DETECTOR_BACKEND}: {e}"
        )
        frame_entry = {"frame": frame_count}
        for emotion_key in emotions:
            frame_entry[emotion_key] = 0.0
        return (frame_count, frame_entry)


# --- Main Execution Block ---
if __name__ == "__main__":
    # Set multiprocessing start method once
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # print("Multiprocessing start method already set or can't be forced.")
        pass  # If it's already set or can't be changed, continue.

    print(f"Using {NUM_PROCESSES} processes for analysis of each video.")
    print(f"Using face detector backend: {DETECTOR_BACKEND}")

    # --- Step 0: Setup Directories ---
    source_dir_path = pathlib.Path(source_videos_directory)
    if not source_dir_path.is_dir():
        print(f"Error: Source videos directory not found: {source_videos_directory}")
        exit()

    analysis_output_parent_path = pathlib.Path(analysis_output_parent_directory)
    analysis_output_parent_path.mkdir(parents=True, exist_ok=True)
    print(f"Analysis folders will be created in: {analysis_output_parent_path}")

    # --- Find Video Files (MP4 only) ---
    video_files = []
    print(f"Searching for MP4 videos in: {source_videos_directory}")
    for item in source_dir_path.iterdir():
        if item.is_file() and item.suffix.lower() in video_extensions:
            video_files.append(item)

    if not video_files:
        print(f"No MP4 video files found in {source_videos_directory}.")
        exit()

    print(f"Found {len(video_files)} videos to process.")
    script_start_time = time.time()

    # --- Process Each Video ---
    for video_file in video_files:
        print("\n" + "=" * 50)
        print(f"Processing video: {video_file.name}")
        start_time_video = time.time()

        # --- Create Analysis Output Folder for the current video ---
        analysis_folder_name = video_file.stem + "_analyze"
        analysis_folder_path = analysis_output_parent_path / analysis_folder_name
        analysis_folder_path.mkdir(parents=True, exist_ok=True)
        print(
            f"  Analysis output for '{video_file.name}' will be saved in: {analysis_folder_path}"
        )

        # --- Step 1: Load the Video (for reading frames for analysis) ---
        cap_analyze = cv2.VideoCapture(str(video_file))
        if not cap_analyze.isOpened():
            print(f"  Error: Could not open video {video_file.name}. Skipping.")
            continue

        total_frames = int(cap_analyze.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap_analyze.get(cv2.CAP_PROP_FPS)
        print(f"  '{video_file.name}': Total frames={total_frames}, FPS={fps:.2f}")

        frames_to_process = []
        frame_count_read = (
            0  # Renamed to avoid confusion with frame_count in analyze_single_frame
        )
        while True:
            ret, frame = cap_analyze.read()
            if not ret:
                break
            frame_count_read += 1
            frames_to_process.append((frame_count_read, frame.copy()))
        cap_analyze.release()

        if not frames_to_process:
            print(f"  No frames read from '{video_file.name}'. Skipping this video.")
            continue
        print(
            f"  Read {len(frames_to_process)} frames. Starting parallel emotion analysis using '{DETECTOR_BACKEND}'..."
        )

        # --- Step 2-5 (Parallel): Analyze frames ---
        analysis_start_time = time.time()

        with mp.Pool(processes=NUM_PROCESSES) as pool:
            # Adjust chunksize: A larger chunksize can be more efficient for very many short tasks
            # but smaller can give better load balancing.
            # Heuristic: at least 1, and aim for a few chunks per process.
            num_tasks = len(frames_to_process)
            chunksize = max(1, num_tasks // (NUM_PROCESSES * 4))
            if (
                num_tasks < NUM_PROCESSES * 4
            ):  # Ensure small tasks get processed quickly too
                chunksize = max(1, num_tasks // NUM_PROCESSES)

            results_iterator = pool.imap_unordered(
                analyze_single_frame,
                frames_to_process,
                chunksize=chunksize,
            )

            processed_results = []
            total_frames_to_analyze = len(frames_to_process)
            for i, result in enumerate(results_iterator):
                processed_results.append(result)
                if (i + 1) % 200 == 0 or (i + 1) == total_frames_to_analyze:
                    print(f"    Analyzed {i + 1}/{total_frames_to_analyze} frames...")

        analysis_end_time = time.time()
        print(
            f"  Parallel analysis finished in {analysis_end_time - analysis_start_time:.2f} seconds."
        )

        # --- Step 6: Process and Store Data (Analysis Data) ---
        if not processed_results:
            print(
                f"  No results returned from analysis for '{video_file.name}'. Skipping CSV and plot generation."
            )
            continue

        processed_results.sort(key=lambda x: x[0])  # Sort by frame number
        emotion_data_list = [
            item[1]
            for item in processed_results
            if item is not None
            and item[1] is not None  # Ensure item and its data are not None
        ]

        if not emotion_data_list:
            print(
                f"  No valid emotion data collected for '{video_file.name}'. Skipping CSV and plot generation."
            )
            continue

        df = pd.DataFrame(emotion_data_list)
        if df.empty:
            print(
                f"  DataFrame is empty for '{video_file.name}'. Skipping CSV and plot generation."
            )
            continue

        csv_output_path = analysis_folder_path / "emotion_data.csv"
        df.to_csv(csv_output_path, index=False)
        print(f"  Emotion data saved to {csv_output_path}")

        # --- Step 7: Plot and Save Analysis Results ---
        print(f"  Generating plots for '{video_file.name}'...")
        plot_generation_start_time = time.time()

        plt.figure(figsize=(15, 7))
        for emotion_key in emotions:
            if emotion_key in df.columns:
                plt.plot(df["frame"], df[emotion_key], label=emotion_key.capitalize())
        plt.xlabel("Frame Number")
        plt.ylabel("Emotion Score (0-100)")
        plt.title(
            f"Facial Emotion Over Time ({video_file.name}) - Detector: {DETECTOR_BACKEND}"
        )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        all_plot_path = (
            analysis_folder_path / f"all_emotions_plot_{DETECTOR_BACKEND}.png"
        )
        try:
            plt.savefig(all_plot_path)
            print(f"  'All emotions' plot saved to {all_plot_path}")
        except Exception as e_plot:
            print(f"  Error saving all_emotions_plot: {e_plot}")
        plt.close()

        for emotion_key in emotions:
            if emotion_key in df.columns:
                plt.figure(figsize=(10, 5))
                plt.plot(
                    df["frame"],
                    df[emotion_key],
                    label=emotion_key.capitalize(),
                    color="blue",
                )
                plt.xlabel("Frame Number")
                plt.ylabel(f"{emotion_key.capitalize()} Score (0-100)")
                plt.title(
                    f"{emotion_key.capitalize()} Emotion Over Time ({video_file.name}) - Detector: {DETECTOR_BACKEND}"
                )
                plt.grid(True)
                plt.tight_layout()
                individual_plot_path = (
                    analysis_folder_path
                    / f"{emotion_key}_emotion_plot_{DETECTOR_BACKEND}.png"
                )
                try:
                    plt.savefig(individual_plot_path)
                except Exception as e_plot_ind:
                    print(f"  Error saving {emotion_key}_emotion_plot: {e_plot_ind}")
                plt.close()
        print(
            f"  Plots generated in {time.time() - plot_generation_start_time:.2f} seconds."
        )

        end_time_video = time.time()
        print(
            f"  Finished processing video '{video_file.name}' in {end_time_video - start_time_video:.2f} seconds."
        )

    print("\n" + "=" * 50)
    print(f"All videos processed in {time.time() - script_start_time:.2f} seconds.")
