# ─── THREAD / TF CONFIG: MUST COME BEFORE DEEPFACE / TENSORFLOW IMPORTS ───────
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Optional:
# Use this if you want to force a specific GPU, e.g. GPU 0.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2

cv2.setNumThreads(0)

import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import pathlib
import traceback

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_DIR = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\2. Video Facial Emotion Recognition (VFER)\Dataset\DFEW_face"

OUTPUT_DIR = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\2. Video Facial Emotion Recognition (VFER)\Dataset\DFEW_face_analyze"

# IMPORTANT:
# Use "skip" only if videos are already face-cropped.
# Since your input folder is DFEW_face, this is likely appropriate.
DETECTOR_BACKEND = "skip"

# With detector_backend="skip", alignment is not useful.
ALIGN_FACE = False

GENERATE_PLOTS = True

# GPU recommendation:
# TensorFlow/DeepFace usually grabs most GPU memory, so use 1 worker.
# If you have a very large GPU and want to experiment, try 2.
NUM_WORKERS = 16

# If you later run CPU-only, try:
# NUM_WORKERS = min(4, max(1, mp.cpu_count() - 1))

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# DeepFace is imported lazily inside each worker process.
DeepFace = None


# ─── WORKER INITIALIZER ───────────────────────────────────────────────────────
def _worker_init(backend: str, align_face: bool) -> None:
    """
    Called once per worker process.

    This:
    1. Configures TensorFlow GPU memory growth.
    2. Imports DeepFace inside the worker.
    3. Warms up the emotion model once.
    """
    global DeepFace
    global _BACKEND
    global _ALIGN_FACE

    _BACKEND = backend
    _ALIGN_FACE = align_face

    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception:
                    pass

            print(f"[GPU] Worker PID {os.getpid()} sees {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"      {gpu}")
        else:
            print(
                f"[WARN] Worker PID {os.getpid()} does not see a GPU. Running on CPU."
            )

    except Exception as e:
        print(f"[WARN] Could not configure TensorFlow GPU memory growth: {e}")

    from deepface import DeepFace as _DeepFace

    DeepFace = _DeepFace

    # Warm-up model once so the first real frame does not pay initialization cost.
    try:
        import numpy as np

        dummy = np.zeros((224, 224, 3), dtype="uint8")

        DeepFace.analyze(
            dummy,
            actions=["emotion"],
            detector_backend=_BACKEND,
            enforce_detection=False,
            align=_ALIGN_FACE,
            silent=True,
        )

        print(f"[INIT] Worker PID {os.getpid()} warmed up DeepFace.")

    except Exception as e:
        print(f"[WARN] DeepFace warm-up failed in PID {os.getpid()}: {e}")


# ─── FRAME ANALYSIS ───────────────────────────────────────────────────────────
def analyze_single_frame(frame_count: int, frame) -> dict:
    """
    Analyze one frame and return one CSV row.
    Keeps the same output columns as your original script.
    """
    entry = {"frame": frame_count}

    try:
        results = DeepFace.analyze(
            frame,
            actions=["emotion"],
            detector_backend=_BACKEND,
            enforce_detection=False,
            align=_ALIGN_FACE,
            silent=True,
        )

        # DeepFace may return either a list or a dict depending on version.
        if isinstance(results, list) and len(results) > 0:
            emotions_in_frame = results[0].get("emotion", {})
        elif isinstance(results, dict):
            emotions_in_frame = results.get("emotion", {})
        else:
            emotions_in_frame = {}

    except Exception as e:
        print(f"  [WARN] Frame {frame_count} PID {os.getpid()}: {e}")
        emotions_in_frame = {}

    for emotion in EMOTIONS:
        entry[emotion] = float(emotions_in_frame.get(emotion, 0.0))

    return entry


# ─── VIDEO PROCESSING ─────────────────────────────────────────────────────────
def process_video_task(args: tuple) -> tuple:
    """
    One worker processes one whole video.

    This is faster than sending every frame through multiprocessing because:
    - frames stay inside the same process
    - no frame pickling / IPC overhead
    - GPU model stays warm inside that process
    """
    video_path_str, output_dir_str = args

    video_path = pathlib.Path(video_path_str)
    output_dir = pathlib.Path(output_dir_str)

    class_name = video_path.parent.name
    video_output_dir = output_dir / class_name / (video_path.stem + "_analyze")
    csv_path = video_output_dir / "emotion_data.csv"

    if csv_path.exists():
        return ("skip", class_name, video_path.name, 0, 0.0, "already analyzed")

    video_output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ("error", class_name, video_path.name, 0, 0.0, "could not open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return ("error", class_name, video_path.name, 0, fps, "zero frames")

    records = [None] * total_frames

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Keep your original speed optimization:
            # Downscale very large frames to width 480.
            # This usually does not hurt emotion recognition if faces remain clear.
            h, w = frame.shape[:2]
            if w > 480:
                scale = 480 / w
                frame = cv2.resize(
                    frame,
                    (480, int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            entry = analyze_single_frame(frame_count, frame)

            # frame_count starts at 1, list index starts at 0.
            if frame_count <= total_frames:
                records[frame_count - 1] = entry
            else:
                records.append(entry)

    except Exception as e:
        cap.release()
        tb = traceback.format_exc()
        return ("error", class_name, video_path.name, frame_count, fps, f"{e}\n{tb}")

    cap.release()

    records = [r for r in records if r is not None]

    if not records:
        return ("error", class_name, video_path.name, 0, fps, "no analysis results")

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

    if GENERATE_PLOTS:
        _save_plots(df, video_path.name, video_output_dir)

    return ("ok", class_name, video_path.name, len(records), fps, str(csv_path))


# ─── PLOTTING ─────────────────────────────────────────────────────────────────
def _save_plots(df: pd.DataFrame, video_name: str, output_dir: pathlib.Path) -> None:
    plt.figure(figsize=(15, 7))

    for emotion in EMOTIONS:
        if emotion in df.columns:
            plt.plot(df["frame"], df[emotion], label=emotion.capitalize())

    plt.xlabel("Frame Number")
    plt.ylabel("Emotion Score (0–100)")
    plt.title(f"Facial Emotions — {video_name} [{DETECTOR_BACKEND}]")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    try:
        plt.savefig(output_dir / f"all_emotions_{DETECTOR_BACKEND}.png")
    except Exception as e:
        print(f"  [WARN] Could not save all_emotions plot: {e}")

    plt.close()

    for emotion in EMOTIONS:
        if emotion not in df.columns:
            continue

        plt.figure(figsize=(10, 5))
        plt.plot(df["frame"], df[emotion], label=emotion.capitalize())
        plt.xlabel("Frame Number")
        plt.ylabel(f"{emotion.capitalize()} Score (0–100)")
        plt.title(f"{emotion.capitalize()} — {video_name} [{DETECTOR_BACKEND}]")
        plt.grid(True)
        plt.tight_layout()

        try:
            plt.savefig(output_dir / f"{emotion}_{DETECTOR_BACKEND}.png")
        except Exception as e:
            print(f"  [WARN] Could not save {emotion} plot: {e}")

        plt.close()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    input_root = pathlib.Path(INPUT_DIR)
    output_root = pathlib.Path(OUTPUT_DIR)

    if not input_root.is_dir():
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        raise SystemExit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        list(input_root.rglob("*.mp4")),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )

    if not video_files:
        print(f"No MP4 files found under {INPUT_DIR}")
        raise SystemExit(1)

    total = len(video_files)

    print("=" * 70)
    print(f"Found videos      : {total}")
    print(f"Detector backend  : {DETECTOR_BACKEND}")
    print(f"Align face        : {ALIGN_FACE}")
    print(f"Workers           : {NUM_WORKERS}")
    print(f"Output directory  : {OUTPUT_DIR}")
    print("=" * 70)
    print()

    script_start = time.time()
    done = 0
    skipped = 0
    failed = 0

    tasks = [(str(video_path), str(output_root)) for video_path in video_files]

    with mp.Pool(
        processes=NUM_WORKERS,
        initializer=_worker_init,
        initargs=(DETECTOR_BACKEND, ALIGN_FACE),
    ) as pool:

        for (
            status,
            class_name,
            video_name,
            n_frames,
            fps,
            message,
        ) in pool.imap_unordered(
            process_video_task,
            tasks,
            chunksize=1,
        ):
            done += 1

            if status == "ok":
                print(
                    f"[{done}/{total}] [OK]   {class_name}/{video_name} — "
                    f"{n_frames} frames, FPS={fps:.1f}"
                )

            elif status == "skip":
                skipped += 1
                print(f"[{done}/{total}] [SKIP] {class_name}/{video_name}")

            else:
                failed += 1
                print(f"[{done}/{total}] [ERROR] {class_name}/{video_name}")
                print(f"          {message}")

            elapsed_total = time.time() - script_start
            rate = done / elapsed_total if elapsed_total > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0

            print(
                f"          Progress: {rate:.3f} vid/s | " f"ETA {eta / 60:.1f} min\n"
            )

    total_time_min = (time.time() - script_start) / 60

    print("=" * 70)
    print(f"Finished.")
    print(f"Processed : {done - skipped - failed}")
    print(f"Skipped   : {skipped}")
    print(f"Failed    : {failed}")
    print(f"Total     : {total}")
    print(f"Time      : {total_time_min:.1f} min")
    print(f"Results   : {OUTPUT_DIR}")
    print("=" * 70)
