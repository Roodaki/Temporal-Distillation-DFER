import argparse
import os
from typing import Optional
import cv2
from glob import glob
import mediapipe as mp
import multiprocessing
from multiprocessing import Pool
import time

PARAMS = {
    "detection_confidence": 0.5,
    "model_selection": 0,
    "output_size": (224, 224),
    "padding_ratio": 0.1,
    "codec": "mp4v",
    "fps": None,
    "show_progress": False,
    "skip_missing_faces": True,
    "min_output_frames": 1,
}


def collect_tasks(input_root: str, output_root: str) -> list:
    tasks = []
    for class_name in sorted(os.listdir(input_root)):
        class_input_dir = os.path.join(input_root, class_name)
        class_output_dir = os.path.join(output_root, class_name)

        if not os.path.isdir(class_input_dir):
            continue

        os.makedirs(class_output_dir, exist_ok=True)

        videos = [
            f
            for f in glob(os.path.join(class_input_dir, "*.mp4"))
            if "_face.mp4" not in f
        ]

        # Sort by file size descending — process longest videos first
        # so large jobs don't end up at the tail and stall completion
        videos.sort(key=lambda f: os.path.getsize(f), reverse=True)

        for vf in videos:
            tasks.append((vf, class_output_dir, PARAMS))

        print(f"  [{class_name}] {len(videos)} videos queued")

    return tasks


def process_single_video(args: tuple) -> dict:
    """
    Worker function. Returns a result dict instead of raising,
    so failures don't kill the whole pool.
    """
    input_path, output_dir, params = args
    input_filename = os.path.basename(input_path)
    base_name = os.path.splitext(input_filename)[0]
    output_path = os.path.join(output_dir, f"{base_name}.mp4")

    # Skip if already processed with real content
    if os.path.exists(output_path) and os.path.getsize(output_path) > 10_000:
        return {"status": "skipped", "file": input_filename}

    try:
        frames_written = detect_and_crop_face(
            input_video_path=input_path,
            output_video_path=output_path,
            **{k: v for k, v in params.items() if k != "min_output_frames"},
        )

        min_frames = params.get("min_output_frames", 1)

        if frames_written < min_frames:
            if os.path.exists(output_path):
                os.remove(output_path)
            return {
                "status": "no_face",
                "file": input_filename,
                "frames": frames_written,
            }

        return {"status": "ok", "file": input_filename, "frames": frames_written}

    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        return {"status": "error", "file": input_filename, "error": str(e)}


def detect_and_crop_face(
    input_video_path: str,
    output_video_path: str,
    detection_confidence: float = 0.5,
    model_selection: int = 0,
    output_size: tuple = (224, 224),
    padding_ratio: float = 0.1,
    codec: str = "mp4v",
    fps: Optional[float] = None,
    show_progress: bool = False,
    skip_missing_faces: bool = True,
) -> int:
    """Returns the number of frames successfully written."""
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=model_selection, min_detection_confidence=detection_confidence
    )

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fps = fps if fps is not None else (input_fps if input_fps > 0 else 25.0)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, output_size)

    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter for: {output_video_path}")

    frames_written = 0

    try:
        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                ih, iw = frame.shape[:2]

                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)

                pad_x = int(w * padding_ratio)
                pad_y = int(h * padding_ratio)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(iw, x + w + pad_x), min(ih, y + h + pad_y)

                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    if skip_missing_faces:
                        continue
                    raise ValueError(f"Invalid face crop at frame {frame_num}")

                out.write(cv2.resize(cropped, output_size))
                frames_written += 1

                if show_progress:
                    print(f"  Frame {frame_num+1}/{total_frames}", end="\r")
            else:
                if not skip_missing_faces:
                    raise ValueError(f"No face detected at frame {frame_num}")
    finally:
        cap.release()
        out.release()
        face_detection.close()

    return frames_written


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MediaPipe face ROI extraction")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/raw_videos",
        help="ImageNet-style root of raw class-labeled videos.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/cropped_faces",
        help="Where cropped-face videos are written, mirroring the class structure.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    args = get_args()
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir

    print(f"Scanning: {INPUT_DIR}")
    tasks = collect_tasks(INPUT_DIR, OUTPUT_DIR)
    total = len(tasks)
    print(f"\nTotal videos to process: {total}")

    # Use all cores — Pool with imap_unordered gives the best dynamic scheduling:
    # as soon as any worker finishes a video it immediately picks up the next one,
    # no worker ever sits idle while tasks remain in the queue.
    num_workers = multiprocessing.cpu_count()
    print(f"Using {num_workers} workers\n")

    done, ok, skipped, no_face, failed = 0, 0, 0, 0, 0
    start = time.time()

    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(process_single_video, tasks, chunksize=1):
            done += 1
            status = result["status"]

            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            elif status == "no_face":
                no_face += 1
            elif status == "error":
                failed += 1
                print(f"  [ERROR] {result['file']}: {result.get('error')}")

            # Progress every 50 videos
            if done % 50 == 0 or done == total:
                elapsed = time.time() - start
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0
                print(
                    f"  [{done}/{total}] "
                    f"ok={ok} skipped={skipped} no_face={no_face} errors={failed} | "
                    f"{rate:.1f} vid/s | ETA {eta/60:.1f} min"
                )

    elapsed = time.time() - start
    print(f"\n✓ Done in {elapsed/60:.1f} min:")
    print(f"   {ok} processed successfully")
    print(f"   {skipped} skipped (already existed)")
    print(f"   {no_face} deleted (no face detected)")
    print(f"   {failed} errors")
    print(f"\nOutput: {OUTPUT_DIR}")
