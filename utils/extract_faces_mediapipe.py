import os
from typing import Optional
import concurrent
import cv2
from glob import glob
import mediapipe as mp
from concurrent.futures import ProcessPoolExecutor
import multiprocessing


def process_directory(
    input_dir: str, output_dir: str, max_workers: Optional[int] = None
) -> None:
    """
    Process all MP4 videos in a directory with face cropping using multiprocessing

    Args:
        input_dir: Directory containing input videos
        output_dir: Directory to save processed videos
        max_workers: Number of parallel processes (default: CPU count)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all mp4 files in directory
    video_files = glob(os.path.join(input_dir, "*.mp4"))
    video_files = [f for f in video_files if "_face.mp4" not in f]

    print(f"Found {len(video_files)} videos to process")

    # Create process pool
    num_workers = max_workers or multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create list of tasks
        tasks = [
            (
                vf,
                output_dir,
                {
                    "detection_confidence": 0.5,
                    "model_selection": 0,
                    "output_size": (448, 448),
                    "padding_ratio": 0.0,
                    "codec": "mp4v",
                    "fps": None,
                    "show_progress": False,
                    "skip_missing_faces": True,
                },
            )
            for vf in video_files
        ]

        # Submit all tasks
        futures = [executor.submit(process_single_video, *task) for task in tasks]

        # Monitor progress
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    print(f"Completed: {result}")
            except Exception as e:
                print(f"Error processing video: {str(e)}")


def process_single_video(input_path: str, output_dir: str, params: dict) -> str:
    """
    Process a single video with error handling
    Returns the processed filename if successful
    """
    # Create output filename
    input_filename = os.path.basename(input_path)
    base_name = os.path.splitext(input_filename)[0]
    output_path = os.path.join(output_dir, f"{base_name}_face.mp4")

    try:
        print(f"Starting processing: {input_filename}")
        detect_and_crop_face(
            input_video_path=input_path, output_video_path=output_path, **params
        )

        # Verify output
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return f"Success: {input_filename} -> {os.path.basename(output_path)}"
        raise RuntimeError("Output file verification failed")

    except Exception as e:
        # Clean up failed output
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Failed {input_filename}: {str(e)}") from e


def detect_and_crop_face(
    input_video_path: str,
    output_video_path: str,
    detection_confidence: float = 0.5,
    model_selection: int = 0,
    output_size: tuple = (448, 448),
    padding_ratio: float = 0.0,
    codec: str = "mp4v",
    fps: Optional[float] = None,
    show_progress: bool = True,
    skip_missing_faces: bool = True,
) -> None:
    """
    Detects and crops faces from a video using MediaPipe's face detection model.

    Args:
        input_video_path: Path to input video file
        output_video_path: Path to save output video
        detection_confidence: Minimum confidence threshold for face detection (0.0-1.0)
        model_selection: MediaPipe model selection (0=short-range, 1=long-range)
        output_size: Output frame dimensions (width, height)
        padding_ratio: Additional padding around detected face (relative to face size)
        codec: Output video codec (fourcc format)
        fps: Optional output FPS (uses input FPS if None)
        show_progress: Print processing progress
        skip_missing_faces: Skip frames with no detection instead of raising error
    """
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=model_selection, min_detection_confidence=detection_confidence
    )

    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    # Get video properties
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set output FPS and initialize writer
    output_fps = fps if fps is not None else input_fps
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, output_size)

    try:
        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces in RGB frame
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.detections:
                # Get first face (assumed to be main subject)
                bbox = results.detections[0].location_data.relative_bounding_box
                ih, iw = frame.shape[:2]

                # Calculate padded bounding box
                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                w = int(bbox.width * iw)
                h = int(bbox.height * ih)

                pad_x = int(w * padding_ratio)
                pad_y = int(h * padding_ratio)
                x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
                x2, y2 = min(iw, x + w + pad_x), min(ih, y + h + pad_y)

                # Crop and resize face region
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    if skip_missing_faces:
                        continue
                    raise ValueError(f"Invalid face crop in frame {frame_num}")

                resized = cv2.resize(cropped, output_size)
                out.write(resized)

                if show_progress:
                    print(f"Processed frame {frame_num+1}/{total_frames}", end="\r")
            else:
                if not skip_missing_faces:
                    raise ValueError(f"No face detected in frame {frame_num}")
    finally:
        cap.release()
        out.release()


if __name__ == "__main__":
    # Example usage
    process_directory(
        input_dir=r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Dataset\rotated",
        output_dir=r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Dataset\rotated_face",
    )
