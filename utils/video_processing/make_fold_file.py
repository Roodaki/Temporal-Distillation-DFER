"""
Build a fold-assignment CSV ("video,fold") for --fold_file in the training
scripts from per-fold test-split lists, e.g. the official DFEW five-fold split
(test(single-labeled)/set_1.csv ... set_5.csv).

Each input CSV lists the videos of one test fold. The fold id is taken from a
"set_<k>" pattern in the file name when present, otherwise from the position
of the file on the command line (1-based).

Example:
    python utils/video_processing/make_fold_file.py \
        --test_csvs ./DFEW/test_single_labeled/set_{1,2,3,4,5}.csv \
        --output ./data/dfew_folds.csv
"""

import argparse
import csv
import os
import re

VIDEO_COLUMN_CANDIDATES = ("video_name", "video", "name", "filename")


def read_videos(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
    if not rows:
        return []
    header = [h.strip().lower() for h in rows[0]]
    for candidate in VIDEO_COLUMN_CANDIDATES:
        if candidate in header:
            col = header.index(candidate)
            return [row[col].strip() for row in rows[1:] if len(row) > col]
    # No recognizable header: first column, keep the first row unless it is
    # clearly a header (non-numeric while the rest are numeric).
    values = [row[0].strip() for row in rows]
    if len(values) > 1 and not values[0].isdigit() and values[1].isdigit():
        values = values[1:]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--test_csvs", nargs="+", required=True, help="One CSV per test fold.")
    parser.add_argument("--output", required=True, help="Output CSV with columns video,fold.")
    args = parser.parse_args()

    assignment = {}
    for position, path in enumerate(args.test_csvs, start=1):
        m = re.search(r"set_(\d+)", os.path.basename(path))
        fold = int(m.group(1)) if m else position
        videos = read_videos(path)
        for video in videos:
            if video in assignment and assignment[video] != fold:
                raise SystemExit(
                    f"Video '{video}' appears in test folds {assignment[video]} and {fold}."
                )
            assignment[video] = fold
        print(f"Fold {fold}: {len(videos)} videos from {path}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "fold"])
        for video, fold in assignment.items():
            writer.writerow([video, fold])
    print(f"Wrote {len(assignment)} fold assignments to {args.output}")


if __name__ == "__main__":
    main()
