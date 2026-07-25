import argparse
import pathlib
from collections import defaultdict

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}

# The three passes run_max_emotion_pass / run_min_neutral_pass / run_random_pass
# write to <output_dir>/<pass_name>/<class>/...
PASS_NAMES = ["max_emotion", "min_neutral", "random"]


def count_clips_per_class(pass_root: pathlib.Path) -> dict:
    """Count video files per immediate child folder (class) under pass_root."""
    counts = defaultdict(int)
    if not pass_root.is_dir():
        return counts
    for class_dir in sorted(pass_root.iterdir()):
        if not class_dir.is_dir():
            continue
        n = sum(
            1
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        counts[class_dir.name] = n
    return counts


def print_table(title: str, counts: dict):
    print(f"\n=== {title} ===")
    if not counts:
        print("  (no data found)")
        return
    max_name_len = max(len(name) for name in counts) if counts else 10
    total = 0
    for class_name in sorted(counts):
        n = counts[class_name]
        total += n
        print(f"  {class_name.ljust(max_name_len)} : {n}")
    print(f"  {'TOTAL'.ljust(max_name_len)} : {total}")


def main():
    parser = argparse.ArgumentParser(
        description="Count extracted clips per class for each pass under an output directory "
        "produced by the temporal distillation script."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/roodaki/projects/DFEW/data/dfew/DFEW_face_trimmed_8",
        help="Root output directory (the same --output_dir passed to the distillation script).",
    )
    parser.add_argument(
        "--passes",
        type=str,
        nargs="+",
        default=PASS_NAMES,
        choices=PASS_NAMES,
        help=f"Which pass subfolders to inspect (default: all of {PASS_NAMES}).",
    )
    args = parser.parse_args()

    output_root = pathlib.Path(args.output_dir)
    if not output_root.is_dir():
        print(f"Error: output_dir not found: {output_root}")
        return

    all_pass_counts = {}
    for pass_name in args.passes:
        pass_root = output_root / pass_name
        counts = count_clips_per_class(pass_root)
        all_pass_counts[pass_name] = counts
        print_table(f"{pass_name} ({pass_root})", counts)

    # Combined totals across all inspected passes, per class.
    if len(args.passes) > 1:
        combined = defaultdict(int)
        for counts in all_pass_counts.values():
            for class_name, n in counts.items():
                combined[class_name] += n
        print_table("COMBINED (all inspected passes)", combined)


if __name__ == "__main__":
    main()