import argparse
import cv2
import os
import pathlib
import time
import numpy as np
import multiprocessing as mp

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]
NUM_SEGMENTS_TO_SAMPLE = 1
FOURCC_CODEC = cv2.VideoWriter_fourcc(*"mp4v")
OUTPUT_VIDEO_EXTENSION = ".mp4"


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


def parse_saved_segment_count(result_summary: str) -> int:
    try:
        return int(result_summary.split("Saved ")[1].split(" ")[0])
    except (IndexError, ValueError):
        return 0


# --- Worker Function ---
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

    class_output_dir = trimmed_output_root / relative_class_dir
    class_output_dir.mkdir(parents=True, exist_ok=True)

    cap_props = cv2.VideoCapture(str(video_path))
    if not cap_props.isOpened():
        return f"{relative_class_dir}/{video_stem}: Error - Could not open video."

    fps = cap_props.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap_props.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_props.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_props.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_props.release()

    if fps <= 0 or frame_width <= 0 or frame_height <= 0:
        return f"{relative_class_dir}/{video_stem}: Error - Invalid video properties."

    if total_frames < trim_window_size:
        return (
            f"{relative_class_dir}/{video_stem}: Skipped - Video too short "
            f"({total_frames} frames) for window {trim_window_size}."
        )

    max_valid_start_index = total_frames - trim_window_size
    actual_num_to_sample = min(num_segments, max_valid_start_index + 1)

    try:
        selected_start_indices = np.random.choice(
            max_valid_start_index + 1, actual_num_to_sample, replace=False
        )
    except ValueError:
        return f"{relative_class_dir}/{video_stem}: Error generating random indices."

    selected_start_indices.sort()

    cap_read = cv2.VideoCapture(str(video_path))
    if not cap_read.isOpened():
        return f"{relative_class_dir}/{video_stem}: Error - Could not re-open video for reading."

    segments_saved_count = 0

    for i, start_frame_idx in enumerate(selected_start_indices, start=1):
        output_name = f"{video_stem}_rnd_{i}{output_video_extension}"
        output_file_path = class_output_dir / output_name

        out_writer = cv2.VideoWriter(
            str(output_file_path), fourcc_codec, fps, (frame_width, frame_height)
        )

        if not out_writer.isOpened():
            print(
                f"[{pid}-{relative_class_dir}/{video_stem}] Failed to initialize writer for {output_name}"
            )
            continue

        cap_read.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame_idx))

        segment_ok = True
        for _ in range(trim_window_size):
            ret, frame = cap_read.read()
            if not ret:
                segment_ok = False
                break
            out_writer.write(frame)

        out_writer.release()

        if segment_ok:
            segments_saved_count += 1
        else:
            if output_file_path.exists():
                try:
                    os.remove(output_file_path)
                except Exception as e:
                    print(
                        f"[{pid}-{relative_class_dir}/{video_stem}] Error removing incomplete file {output_file_path}: {e}"
                    )

    cap_read.release()

    duration = time.time() - start_time
    return (
        f"{relative_class_dir}/{video_stem}: Saved {segments_saved_count} random segments "
        f"in {duration:.2f}s."
    )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random temporal trimming baseline")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/cropped_faces",
        help="ImageNet-style root of cropped-face videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/distilled_clips_random",
        help="Where randomly trimmed clips are written.",
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        default=16,
        help="Number of frames per random clip (e.g. 16 for ViViT, 8 for TimeSformer).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = get_args()
    TRIM_WINDOW_SIZE = args.clip_length

    start_script = time.time()

    source_root = pathlib.Path(args.input_dir)
    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    if not source_root.is_dir():
        print(f"Error: Source not found: {source_root}")
        raise SystemExit(1)

    video_files = collect_videos_imagenet(source_root, VIDEO_EXTENSIONS)

    if not video_files:
        print(f"No video files found under ImageNet-style root: {source_root}")
        raise SystemExit(1)

    print(f"Found {len(video_files)} videos under: {source_root}")
    print(f"Outputting to: {output_root}")
    print(
        f"Configuration: {NUM_SEGMENTS_TO_SAMPLE} random segments of size {TRIM_WINDOW_SIZE} per video."
    )

    tasks = [
        (
            str(video_path),
            str(source_root),
            str(output_root),
            TRIM_WINDOW_SIZE,
            NUM_SEGMENTS_TO_SAMPLE,
            FOURCC_CODEC,
            OUTPUT_VIDEO_EXTENSION,
        )
        for video_path in video_files
    ]

    num_procs = max(1, mp.cpu_count() - 1)
    print(f"Starting pool with {num_procs} processes...")

    with mp.Pool(processes=num_procs) as pool:
        results = pool.map(process_single_video_random_worker, tasks)

    success_count = 0
    total_segments = 0

    for res in results:
        print(f"Result: {res}")
        if "Saved" in res:
            success_count += 1
            total_segments += parse_saved_segment_count(res)

    print("-" * 50)
    print(f"Processed {success_count}/{len(video_files)} videos successfully.")
    print(f"Total random clips generated: {total_segments}")
    print(f"Total time: {time.time() - start_script:.2f}s")
