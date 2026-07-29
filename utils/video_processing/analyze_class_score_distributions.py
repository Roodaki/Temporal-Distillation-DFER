"""
analyze_class_score_distributions.py

Diagnostic tool: for every pre-analyzed video (the *_analyze folders produced
by analyze_videos.py, same input the distillation preprocessing script reads
via --logs), compute the single BEST sliding-window score per video for:

  - the "max_emotion" metric: max of the rolling-mean target-emotion score
    (the same quantity the preprocessing script calls best_score_this_video
    in score_single_video_max_emotion_worker)
  - the "min_neutral" metric: min of the rolling-mean neutral score
    (same quantity in score_single_video_min_neutral_worker)

Then aggregate those per-video best scores BY CLASS and report summary
statistics + a box plot, so you can see whether a class (e.g. disgust) is
weak because its best available window is inherently low-signal upstream
(a data/FER-confidence problem), vs. everything else being a downstream
training/modeling problem.

NOTE on total_frames: the original preprocessing script uses the *video's*
actual frame count (via cv2) to cap the valid window range, since the CSV
and the video can very rarely be off by a frame or two. This script uses
len(df) directly instead, to avoid opening thousands of video files just for
a diagnostic pass. This is a proxy, not exact, but the two are equal in the
overwhelming majority of cases and won't change the shape of the class-level
distribution comparison.

Usage:
    python analyze_class_score_distributions.py \
        --logs /path/to/pre_analyzed_root \
        --clip_length 8 \
        --output_dir ./score_distribution_report
"""

import argparse
import multiprocessing as mp
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_TO_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}
NEUTRAL_EMOTION_COLUMN = "neutral"


def find_analysis_folders(pre_analyzed_root: pathlib.Path) -> list[pathlib.Path]:
    return sorted(p for p in pre_analyzed_root.rglob("*_analyze") if p.is_dir())


def best_window_score(df: pd.DataFrame, column: str, window: int, higher_is_better: bool):
    """Best rolling-mean window score for `column`. This is equivalent to
    `non_overlapping_candidates[0]["mean_score"]` in the preprocessing
    script's worker functions, since the top-ranked candidate (rank 1) is
    never affected by the overlap-suppression step."""
    if len(df) < window or column not in df.columns:
        return None
    rolling = df[column].rolling(window=window).mean().dropna()
    if rolling.empty:
        return None
    return float(rolling.max() if higher_is_better else rolling.min())


def score_one_video(task_args):
    analysis_folder_str, pre_analyzed_root_str, clip_length = task_args
    analysis_folder = pathlib.Path(analysis_folder_str)
    pre_analyzed_root = pathlib.Path(pre_analyzed_root_str)

    video_stem = analysis_folder.name.replace("_analyze", "")
    video_class = analysis_folder.parent.name.lower()

    try:
        relative_class_dir = analysis_folder.parent.relative_to(pre_analyzed_root)
    except ValueError:
        return None

    csv_path = analysis_folder / "emotion_data.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    if df.empty or "frame" not in df.columns:
        return None

    target_emotion = CLASS_TO_EMOTION_MAP.get(video_class)

    max_emotion_score = None
    if target_emotion is not None:
        max_emotion_score = best_window_score(df, target_emotion, clip_length, higher_is_better=True)

    min_neutral_score = best_window_score(df, NEUTRAL_EMOTION_COLUMN, clip_length, higher_is_better=False)

    return {
        "video_key": f"{relative_class_dir}/{video_stem}",
        "class_name": video_class,
        "max_emotion_best_score": max_emotion_score,
        "min_neutral_best_score": min_neutral_score,
        "num_csv_rows": len(df),
    }


def summarize(df: pd.DataFrame, score_column: str, label: str):
    valid = df.dropna(subset=[score_column])
    if valid.empty:
        print(f"\n[{label}] No valid scores found.")
        return None

    summary = (
        valid.groupby("class_name")[score_column]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .sort_values("50%", ascending=(label == "min_neutral"))
    )
    print(f"\n=== {label}: best-window score distribution per class ===")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))

    ranked = summary.index.tolist()
    print(f"\nClasses ranked from {'weakest' if label == 'max_emotion' else 'best-separated'} "
          f"to {'strongest' if label == 'max_emotion' else 'least-separated'} "
          f"median best-window score:")
    for rank, cls in enumerate(ranked, start=1):
        median_val = summary.loc[cls, "50%"]
        print(f"  {rank}. {cls}: median={median_val:.3f}")

    return summary


def plot_distributions(df: pd.DataFrame, score_column: str, label: str, output_dir: pathlib.Path):
    valid = df.dropna(subset=[score_column])
    if valid.empty:
        return

    order = (
        valid.groupby("class_name")[score_column]
        .median()
        .sort_values(ascending=(label != "min_neutral"))
        .index.tolist()
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=valid, x="class_name", y=score_column, order=order)
    sns.stripplot(data=valid, x="class_name", y=score_column, order=order,
                   color="black", alpha=0.25, size=2, jitter=True)
    plt.title(f"Per-video best-window score by class ({label})")
    plt.xlabel("Class")
    plt.ylabel("Best window mean score (0-100 scale)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out_path = output_dir / f"best_window_score_distribution_{label}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved plot: {out_path}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Compare per-class best-window score distributions from emotion_data.csv logs."
    )
    parser.add_argument("--logs", type=str, required=True,
                         help="Root of per-video emotion CSV logs (same as preprocessing script's --logs).")
    parser.add_argument("--clip_length", type=int, required=True,
                         help="Window size in frames (same value used for the distilled clips, e.g. 8 or 16).")
    parser.add_argument("--output_dir", type=str, default="./score_distribution_report",
                         help="Where to write the CSV summary and plots.")
    parser.add_argument("--num_processes", type=int, default=max(1, mp.cpu_count() - 1))
    return parser.parse_args()


def main():
    args = get_args()
    pre_analyzed_root = pathlib.Path(args.logs)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_folders = find_analysis_folders(pre_analyzed_root)
    if not analysis_folders:
        print(f"No '*_analyze' folders found under {pre_analyzed_root}.")
        return

    print(f"Found {len(analysis_folders)} analysis folders. Scoring with {args.num_processes} processes...")

    tasks = [(str(f), str(pre_analyzed_root), args.clip_length) for f in analysis_folders]
    with mp.Pool(processes=args.num_processes) as pool:
        results = pool.map(score_one_video, tasks)

    results = [r for r in results if r is not None]
    if not results:
        print("No usable results (check --clip_length and CSV contents).")
        return

    df = pd.DataFrame(results)

    per_video_csv = output_dir / "per_video_best_scores.csv"
    df.to_csv(per_video_csv, index=False)
    print(f"\nSaved per-video raw scores: {per_video_csv}")

    max_emotion_summary = summarize(df, "max_emotion_best_score", "max_emotion")
    min_neutral_summary = summarize(df, "min_neutral_best_score", "min_neutral")

    if max_emotion_summary is not None:
        max_emotion_summary.to_csv(output_dir / "summary_max_emotion.csv")
    if min_neutral_summary is not None:
        min_neutral_summary.to_csv(output_dir / "summary_min_neutral.csv")

    plot_distributions(df, "max_emotion_best_score", "max_emotion", output_dir)
    plot_distributions(df, "min_neutral_best_score", "min_neutral", output_dir)

    print(f"\nDone. Full report in: {output_dir}")


if __name__ == "__main__":
    main()