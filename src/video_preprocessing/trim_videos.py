import argparse
import cv2
import pandas as pd
import os
import pathlib
import time
import numpy as np
import multiprocessing as mp
from collections import defaultdict

# =========================================================================
# Shared configuration
# =========================================================================

CLASS_TO_EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}

VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov"]
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

NUM_TOP_SEGMENTS_TO_SELECT = 10

# ---- Score gating (max-emotion pass) ----
# Scores in emotion_data.csv are on a 0-100 scale. Emotions like "happy" or
# "neutral" are usually detected with high, sustained confidence, while
# emotions like "disgust" or "surprise" are frequently detected with much
# lower confidence even in a genuine expression, because probability mass
# is spread across all classes. A single fixed absolute floor therefore
# ends up being trivial for strong emotions and nearly unreachable for
# weak ones - exactly the classes that most need their K quota to survive.
#
# Instead, qualification is RELATIVE to the best window found in that
# specific video: a window must reach at least
# RELATIVE_SCORE_FLOOR_FRACTION * (that video's own best window score).
# ABSOLUTE_NOISE_FLOOR is a small absolute backstop so that a window is
# never accepted if it's genuinely ~0 signal (e.g. best window itself is
# essentially noise).
RELATIVE_SCORE_FLOOR_FRACTION = 0.05
ABSOLUTE_NOISE_FLOOR = 0.01

# ---- Score gating (min-neutral pass) ----
# Mirrors the logic above but as a ceiling: a window qualifies only if its
# neutral-score is close to that video's own *lowest* (best/least-neutral)
# window, rather than needing to clear a single fixed global ceiling that
# may be trivial or impossible depending on how "neutral-heavy" a given
# video happens to be.
NEUTRAL_EMOTION_COLUMN = "neutral"
RELATIVE_NEUTRAL_CEILING_SLACK_FRACTION = 0.05
ABSOLUTE_NEUTRAL_CEILING_BACKSTOP = 99.99

NUM_RANDOM_SEGMENTS_TO_SAMPLE = NUM_TOP_SEGMENTS_TO_SELECT

FOURCC_CODEC = cv2.VideoWriter_fourcc(*"mp4v")
OUTPUT_VIDEO_EXTENSION = ".mp4"

# ---- K-strategy modes ----
# K-strategy ONLY applies to the random pass now (see history below). It has
# no effect on max_emotion / min_neutral.
# "fixed"           : a single global K for every class (candidate-pool size
#                      per video before class-wide balancing kicks in).
# "manual_balance"  : per-class K supplied by hand as a hyperparameter.
# "auto_balance"    : per-class K computed automatically from class video
#                      counts so that total *candidate clips per class*
#                      trend toward parity (oversampling minority classes)
#                      before the final class-wide balancing step.
K_STRATEGY_CHOICES = ["fixed", "manual_balance", "auto_balance"]

# ---- Uncapped K for max_emotion / min_neutral (all classes) ----
# max_emotion and min_neutral no longer use K/--k_strategy at all. Every
# class's Phase 1 pulls every threshold-qualifying, non-overlapping window it
# can produce, full stop. Rationale: apply_global_topk_balance()'s
# target_count is min() across all classes' candidate pools - so whichever
# class is smallest effectively sets the ceiling every other class gets
# trimmed down to. Any K cap - even one that's *not* the tightest class - can
# only ever push a class's pool down, never up, so a too-conservative K
# (a bad auto_balance estimate, or an under-set --manual_k) can quietly
# produce a false minimum that global_topk has no way to detect or correct
# (it only trims classes ABOVE target_count, never pads classes below it).
# Removing K entirely for every class removes that failure mode: each
# class's pool becomes its true ceiling (threshold + non-overlap, nothing
# else), target_count becomes the true achievable minimum across classes,
# and every class is synced to exactly that. UNCAPPED_K is a sentinel
# consumed by select_non_overlapping_segments(), where it simply never
# triggers the early-stop, so every qualifying window is kept.
UNCAPPED_K = float("inf")

# ---- Class-wide balancing (final step for max_emotion / min_neutral) ----
# K-strategy controls how large a CANDIDATE pool each video is allowed to
# contribute (oversampling minority classes at the per-video level). But if
# a class has enough videos, even a small per-video K can still add up to
# far more candidates than a minority class can ever produce in total - K
# only pushes candidate counts UP, nothing pulls an oversized class DOWN.
#
# BALANCE_MODE controls what happens after all candidates are scored:
#   "none"        : keep every qualified candidate, exactly as K produced it
#                    (old behaviour - classes stay imbalanced if their
#                    videos collectively out-produce minority classes).
#   "global_topk" : compute target_count = size of the SMALLEST class's
#                    candidate pool (after K-oversampling has done its
#                    best). Any class with MORE candidates than target_count
#                    is trimmed down to its class-wide best `target_count`
#                    candidates by score (highest mean target-emotion score
#                    for max_emotion, lowest mean neutral score for
#                    min_neutral) - pooling across ALL videos in that class,
#                    not per-video. Classes with fewer candidates than
#                    target_count keep everything they have (can't
#                    manufacture data that doesn't exist).
BALANCE_MODE_CHOICES = ["none", "global_topk"]
DEFAULT_BALANCE_MODE = "global_topk"


# =========================================================================
# Shared helpers
# =========================================================================


def find_analysis_folders_imagenet(
    pre_analyzed_root: pathlib.Path,
) -> list[pathlib.Path]:
    """Find every *_analyze directory anywhere under an ImageNet-style analysis root."""
    return sorted(p for p in pre_analyzed_root.rglob("*_analyze") if p.is_dir())


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


def read_video_properties(video_path: pathlib.Path):
    """Open a video just to read fps/width/height/frame-count, then release it."""
    cap_props = cv2.VideoCapture(str(video_path))
    if not cap_props.isOpened():
        return None
    fps = cap_props.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap_props.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap_props.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_props.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_props.release()
    if fps <= 0 or frame_width <= 0 or frame_height <= 0:
        return None
    return fps, frame_width, frame_height, total_frames


def write_segment(
    original_video_file: pathlib.Path,
    output_path: pathlib.Path,
    start_frame_zero_based: int,
    trim_window_size: int,
    fps: float,
    frame_width: int,
    frame_height: int,
    fourcc_codec,
    log_prefix: str,
) -> bool:
    """Read `trim_window_size` frames starting at start_frame_zero_based and write them out.
    Returns True on success, False otherwise (and cleans up any partial file)."""
    out_writer = cv2.VideoWriter(
        str(output_path), fourcc_codec, fps, (frame_width, frame_height)
    )
    if not out_writer.isOpened():
        print(
            f"[{log_prefix}] Error: VideoWriter failed for {output_path.name}. Skipping."
        )
        return False

    cap = cv2.VideoCapture(str(original_video_file))
    segment_ok = True

    if not cap.isOpened():
        print(f"[{log_prefix}] Error: Could not open {original_video_file}.")
        segment_ok = False
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_zero_based)
        for _ in range(trim_window_size):
            ret, frame = cap.read()
            if not ret:
                segment_ok = False
                break
            out_writer.write(frame)
        cap.release()

    out_writer.release()

    if not segment_ok and output_path.exists():
        try:
            os.remove(output_path)
        except Exception as e_rem:
            print(
                f"[{log_prefix}] Error removing incomplete file {output_path}: {e_rem}"
            )

    return segment_ok


def evaluate_sliding_windows(
    df: pd.DataFrame,
    score_column: str,
    trim_window_size: int,
    total_frames_original_video: int,
) -> list[dict]:
    """Compute the mean of `score_column` over every valid sliding window and
    return a list of {df_start_index, mean_score, video_start_frame, video_end_frame}.
    """
    max_start_index_csv = len(df) - trim_window_size
    max_start_index_video = total_frames_original_video - trim_window_size

    rolling_means = df[score_column].rolling(window=trim_window_size).mean()

    evaluated_segments = []
    last_valid_start = min(max_start_index_csv, max_start_index_video)
    for i in range(last_valid_start + 1):
        # rolling mean at position (i + trim_window_size - 1) covers window [i, i+trim_window_size)
        window_mean_score = rolling_means.iloc[i + trim_window_size - 1]
        if pd.isna(window_mean_score):
            continue
        evaluated_segments.append(
            {
                "df_start_index": i,
                "mean_score": window_mean_score,
                "video_start_frame": i + 1,
                "video_end_frame": i + trim_window_size,
            }
        )
    return evaluated_segments


def select_non_overlapping_segments(
    ordered_segments: list[dict],
    num_segments_to_select: int,
) -> list[dict]:
    """Greedy temporal non-max suppression.

    `ordered_segments` must already be sorted by priority (best candidate
    first — e.g. highest/lowest mean score, or a random shuffle). Walk the
    list in that order; keep a candidate only if it shares no frame with any
    segment already selected, i.e. reject it if
    candidate.start <= chosen.end AND candidate.end >= chosen.start.
    Stop once `num_segments_to_select` non-overlapping segments are picked
    or the candidate list is exhausted.
    """
    selected_segments: list[dict] = []
    for candidate in ordered_segments:
        if len(selected_segments) >= num_segments_to_select:
            break
        overlaps_existing = any(
            candidate["video_start_frame"] <= chosen["video_end_frame"]
            and candidate["video_end_frame"] >= chosen["video_start_frame"]
            for chosen in selected_segments
        )
        if not overlaps_existing:
            selected_segments.append(candidate)
    return selected_segments


def save_qualified_segments(
    qualified_segments: list[dict],
    original_video_file: pathlib.Path,
    class_output_dir: pathlib.Path,
    video_stem: str,
    filename_suffix: str,
    trim_window_size: int,
    fps: float,
    frame_width: int,
    frame_height: int,
    fourcc_codec,
    output_video_extension: str,
    log_prefix: str,
) -> int:
    """Write out each qualified segment, returning the number successfully saved."""
    saved_count = 0
    filename_suffix_counter = 0
    for segment_info in qualified_segments:
        filename_suffix_counter += 1
        # video_start_frame is one-based; writer needs zero-based.
        start_frame_zero_based = segment_info["video_start_frame"] - 1

        trimmed_video_name = f"{video_stem}_{filename_suffix}_{filename_suffix_counter}{output_video_extension}"
        trimmed_video_path = class_output_dir / trimmed_video_name

        ok = write_segment(
            original_video_file,
            trimmed_video_path,
            start_frame_zero_based,
            trim_window_size,
            fps,
            frame_width,
            frame_height,
            fourcc_codec,
            log_prefix,
        )

        if ok:
            saved_count += 1
        else:
            filename_suffix_counter -= 1

    return saved_count


# =========================================================================
# K-strategy: per-class candidate-pool budget resolution
# =========================================================================


def parse_manual_k_string(k_string: str) -> dict:
    """Parse 'happy=5,sad=20,angry=15' into {'happy': 5, 'sad': 20, 'angry': 15}."""
    per_class_k = {}
    if not k_string:
        return per_class_k
    for pair in k_string.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"Invalid --manual_k entry '{pair}'. Expected format 'class=K'."
            )
        class_name, k_value_str = pair.split("=", 1)
        class_name = class_name.strip().lower()
        try:
            k_value = int(k_value_str.strip())
        except ValueError:
            raise ValueError(
                f"Invalid K value '{k_value_str}' for class '{class_name}' in --manual_k."
            )
        if k_value <= 0:
            raise ValueError(
                f"K value for class '{class_name}' must be a positive integer, got {k_value}."
            )
        per_class_k[class_name] = k_value
    return per_class_k


def count_videos_per_class_imagenet(
    source_root: pathlib.Path, video_extensions: list[str]
) -> dict:
    """Count videos per top-level class folder under an ImageNet-style root.

    Class is taken to be the direct parent folder name of each video file,
    lower-cased, matching how video_class is derived in the worker
    functions elsewhere in this script.
    """
    suffixes = {ext.lower() for ext in video_extensions}
    counts: dict = {}
    for p in source_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in suffixes:
            class_name = p.parent.name.lower()
            counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def compute_auto_balance_k(
    videos_per_class: dict,
    base_k: int,
) -> dict:
    """Derive a per-class candidate-pool K that pushes total candidates-per-class
    toward parity.

    K_class = round(base_k * max_count / count_class)

    The class with the most videos keeps K == base_k. Classes with fewer
    videos get a proportionally larger K (oversampling), so that
    K_class * count_class ~= base_k * max_count for every class, i.e. the
    expected total number of CANDIDATE clips per class is balanced before
    the final class-wide top-K trim.

    No ceiling is applied here: if a class is severely under-represented,
    its computed K can end up larger than what any individual video can
    actually supply (limited by available non-overlapping windows). In
    that case the worker simply returns as many candidates as it can find;
    the shortfall is a natural consequence of the data, not something
    this function silently caps.
    """
    if not videos_per_class:
        return {}
    max_count = max(videos_per_class.values())
    per_class_k = {}
    for class_name, count in videos_per_class.items():
        if count <= 0:
            continue
        k_value = round(base_k * max_count / count)
        per_class_k[class_name] = max(1, k_value)
    return per_class_k


def resolve_per_class_k(
    k_strategy: str,
    base_k: int,
    manual_k_string: str,
    source_root: pathlib.Path,
    video_extensions: list[str],
) -> dict | None:
    """Return a {class_name: K} dict, or None if k_strategy == 'fixed'
    (meaning: use base_k for every class, exactly like the original script).
    """
    if k_strategy == "fixed":
        return None

    if k_strategy == "manual_balance":
        per_class_k = parse_manual_k_string(manual_k_string)
        if not per_class_k:
            raise ValueError(
                "--k_strategy manual_balance requires --manual_k to be set, "
                "e.g. --manual_k 'happy=5,sad=20,angry=15'."
            )
        return per_class_k

    if k_strategy == "auto_balance":
        videos_per_class = count_videos_per_class_imagenet(source_root, video_extensions)
        if not videos_per_class:
            raise ValueError(
                f"--k_strategy auto_balance could not find any class folders with "
                f"videos under {source_root}."
            )
        per_class_k = compute_auto_balance_k(videos_per_class, base_k)
        print("Auto-balance K per class (base_k = {}):".format(base_k))
        for class_name in sorted(per_class_k):
            print(
                f"    {class_name}: count={videos_per_class[class_name]}, "
                f"K={per_class_k[class_name]}"
            )
        return per_class_k

    raise ValueError(f"Unknown k_strategy '{k_strategy}'.")


def get_k_for_class(
    per_class_k: dict | None,
    class_name: str,
    base_k: int,
) -> int:
    """Look up the K to use for a given class.

    Falls back to base_k if per_class_k is None (fixed strategy) or if the
    class is missing from the per-class map (e.g. not mentioned in
    --manual_k) so nothing silently breaks for unlisted classes.
    """
    if per_class_k is None:
        return base_k
    return per_class_k.get(class_name.lower(), base_k)


def format_k_for_display(k) -> str:
    """Render a K value for log/status messages, special-casing UNCAPPED_K
    so logs read 'top all-available' instead of 'top inf'."""
    return "all-available" if k == UNCAPPED_K else str(k)


# =========================================================================
# Class-wide balancing (final step: trims oversized classes down to the
# smallest class's candidate-pool size, keeping only the globally best
# candidates within each oversized class)
# =========================================================================


def apply_global_topk_balance(
    candidates_by_class: dict,
    balance_mode: str,
    higher_is_better: bool,
) -> dict:
    """Given {class_name: [candidate_dict, ...]}, return a new dict trimmed
    according to balance_mode.

    "none"        : returned unchanged.
    "global_topk" : target_count = size of the smallest class's candidate
                    list. Any class with more than target_count candidates
                    is trimmed to its target_count BEST candidates (pooled
                    across all videos in that class, sorted by
                    candidate["mean_score"], highest-first if
                    higher_is_better else lowest-first). Classes with
                    target_count or fewer candidates are left untouched.

    Empty candidate lists (a class that produced literally zero qualified
    candidates) are ignored when computing target_count, since an empty
    class can't set a meaningful floor for everyone else; such a class
    simply contributes 0 either way.
    """
    if balance_mode == "none":
        return candidates_by_class

    if balance_mode != "global_topk":
        raise ValueError(f"Unknown balance_mode '{balance_mode}'.")

    non_empty_counts = [
        len(candidates) for candidates in candidates_by_class.values() if candidates
    ]
    if not non_empty_counts:
        return candidates_by_class

    target_count = min(non_empty_counts)

    print(f"\nClass-wide balancing target (smallest class pool): {target_count} candidates")

    balanced = {}
    for class_name, candidates in candidates_by_class.items():
        if len(candidates) <= target_count:
            balanced[class_name] = candidates
            print(
                f"    {class_name}: {len(candidates)} candidates <= target, keeping all."
            )
            continue

        sorted_candidates = sorted(
            candidates, key=lambda c: c["mean_score"], reverse=higher_is_better
        )
        balanced[class_name] = sorted_candidates[:target_count]
        print(
            f"    {class_name}: {len(candidates)} candidates > target, "
            f"trimmed to top {target_count} by score."
        )

    return balanced


def group_candidates_by_video(candidates: list[dict]) -> dict:
    """Group a flat list of candidates (each carrying its own video identity
    fields) back into {video_key: [candidate, ...]} so Phase 2 can write all
    surviving segments for a given video in one file-open."""
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["video_key"]].append(candidate)
    return grouped


# =========================================================================
# Max-emotion: Phase 1 (score only, no writing)
# =========================================================================


def score_single_video_max_emotion_worker(task_args_tuple):
    (
        analysis_folder_path_str,
        pre_analyzed_root_str,
        source_videos_root_str,
        class_to_emotion_map,
        video_extensions,
        emotions,
        trim_window_size,
        num_top_segments,
        relative_score_floor_fraction,
        absolute_noise_floor,
    ) = task_args_tuple

    analysis_folder_path = pathlib.Path(analysis_folder_path_str)
    pre_analyzed_root = pathlib.Path(pre_analyzed_root_str)
    source_videos_root = pathlib.Path(source_videos_root_str)

    video_stem = analysis_folder_path.name.replace("_analyze", "")

    try:
        relative_class_dir = analysis_folder_path.parent.relative_to(pre_analyzed_root)
    except ValueError:
        return {
            "status": f"{video_stem}: Error - Analysis folder is not under the analysis root.",
            "class_name": None,
            "candidates": [],
        }

    # In ImageNet-style datasets, the class comes from the folder name, not from the filename.
    video_class = analysis_folder_path.parent.name.lower()
    if video_class not in class_to_emotion_map:
        return {
            "status": f"{relative_class_dir}/{video_stem}: Error - Class folder '{video_class}' not in CLASS_TO_EMOTION_MAP.",
            "class_name": video_class,
            "candidates": [],
        }

    target_emotion = class_to_emotion_map[video_class]
    if target_emotion not in emotions:
        return {
            "status": f"{relative_class_dir}/{video_stem}: Error - Mapped emotion '{target_emotion}' is not valid.",
            "class_name": video_class,
            "candidates": [],
        }

    csv_file_path = analysis_folder_path / "emotion_data.csv"
    if not csv_file_path.exists():
        return {
            "status": f"{relative_class_dir}/{video_stem}: Error - CSV not found.",
            "class_name": video_class,
            "candidates": [],
        }

    try:
        df = pd.read_csv(csv_file_path)
        if df.empty or "frame" not in df.columns:
            return {
                "status": f"{relative_class_dir}/{video_stem}: Error - CSV empty or missing 'frame' column.",
                "class_name": video_class,
                "candidates": [],
            }
        if target_emotion not in df.columns:
            return {
                "status": f"{relative_class_dir}/{video_stem}: Error - Target emotion column '{target_emotion}' not found in CSV.",
                "class_name": video_class,
                "candidates": [],
            }
    except Exception as e:
        return {
            "status": f"{relative_class_dir}/{video_stem}: Error - CSV read failed: {e}",
            "class_name": video_class,
            "candidates": [],
        }

    original_video_file = find_source_video(
        source_videos_root, relative_class_dir, video_stem, video_extensions
    )
    if original_video_file is None:
        return {
            "status": f"{relative_class_dir}/{video_stem}: Error - Original video not found under source class folder.",
            "class_name": video_class,
            "candidates": [],
        }

    props = read_video_properties(original_video_file)
    if props is None:
        return {
            "status": f"{relative_class_dir}/{video_stem}: Error - Could not open/read original video properties.",
            "class_name": video_class,
            "candidates": [],
        }
    fps, frame_width, frame_height, total_frames_original_video = props

    can_trim_based_on_data = len(df) >= trim_window_size
    can_trim_based_on_video = total_frames_original_video >= trim_window_size
    if not (can_trim_based_on_data and can_trim_based_on_video):
        return {
            "status": (
                f"{relative_class_dir}/{video_stem}: Skipping - Not enough frames in CSV "
                f"({len(df)}) or video ({total_frames_original_video}) for window {trim_window_size}."
            ),
            "class_name": video_class,
            "candidates": [],
        }

    all_evaluated_segments = evaluate_sliding_windows(
        df, target_emotion, trim_window_size, total_frames_original_video
    )

    if not all_evaluated_segments:
        return {
            "status": f"{relative_class_dir}/{video_stem}: No valid segments could be evaluated.",
            "class_name": video_class,
            "candidates": [],
        }

    # Highest mean target-emotion score first.
    all_evaluated_segments.sort(key=lambda x: x["mean_score"], reverse=True)
    non_overlapping_candidates = select_non_overlapping_segments(
        all_evaluated_segments, num_top_segments
    )

    if not non_overlapping_candidates:
        return {
            "status": f"{relative_class_dir}/{video_stem}: No non-overlapping candidate segments found.",
            "class_name": video_class,
            "candidates": [],
        }

    # Qualification is relative to THIS VIDEO's own best window, rather than
    # a fixed global floor (see module docstring constants above).
    best_score_this_video = non_overlapping_candidates[0]["mean_score"]
    relative_floor = max(
        relative_score_floor_fraction * best_score_this_video, absolute_noise_floor
    )

    qualified_segments = [
        segment
        for segment in non_overlapping_candidates
        if segment["mean_score"] >= relative_floor
    ]

    video_key = f"{relative_class_dir}/{video_stem}"
    for segment in qualified_segments:
        segment["video_key"] = video_key
        segment["class_name"] = video_class
        segment["video_stem"] = video_stem
        segment["relative_class_dir"] = str(relative_class_dir)
        segment["original_video_file"] = str(original_video_file)
        segment["fps"] = fps
        segment["frame_width"] = frame_width
        segment["frame_height"] = frame_height

    status = (
        f"{video_key}: Scored. {len(qualified_segments)} candidates qualified "
        f"(relative floor {relative_floor:.4f}, best window {best_score_this_video:.4f})."
        if qualified_segments
        else (
            f"{video_key}: No segments from top {format_k_for_display(num_top_segments)} met relative "
            f"score floor {relative_floor:.4f} (best window: {best_score_this_video:.4f})."
        )
    )

    return {
        "status": status,
        "class_name": video_class,
        "candidates": qualified_segments,
    }


# =========================================================================
# Min-neutral: Phase 1 (score only, no writing)
# =========================================================================


def score_single_video_min_neutral_worker(task_args_tuple):
    (
        analysis_folder_path_str,
        pre_analyzed_root_str,
        source_videos_root_str,
        video_extensions,
        neutral_column,
        trim_window_size,
        num_top_segments,
        relative_neutral_ceiling_slack_fraction,
        absolute_neutral_ceiling_backstop,
    ) = task_args_tuple

    analysis_folder_path = pathlib.Path(analysis_folder_path_str)
    pre_analyzed_root = pathlib.Path(pre_analyzed_root_str)
    source_videos_root = pathlib.Path(source_videos_root_str)

    video_stem = analysis_folder_path.name.replace("_analyze", "")

    try:
        relative_class_dir = analysis_folder_path.parent.relative_to(pre_analyzed_root)
    except ValueError:
        return {
            "status": f"{video_stem}: Error - Analysis folder is not under the analysis root.",
            "class_name": None,
            "candidates": [],
        }

    video_class = analysis_folder_path.parent.name.lower()

    csv_file_path = analysis_folder_path / "emotion_data.csv"
    if not csv_file_path.exists():
        return {
            "status": f"{relative_class_dir}/{video_stem}: Error - CSV not found.",
            "class_name": video_class,
            "candidates": [],
        }

    try:
        df = pd.read_csv(csv_file_path)
        if df.empty or "frame" not in df.columns:
            return {
                "status": f"{relative_class_dir}/{video_stem}: Error - CSV empty or missing 'frame' column.",
                "class_name": video_class,
                "candidates": [],
            }
        if neutral_column not in df.columns:
            return {
                "status": f"{relative_class_dir}/{video_stem}: Error - '{neutral_column}' column not found in CSV.",
                "class_name": video_class,
                "candidates": [],
            }
    except Exception as e:
        return {
            "status": f"{relative_class_dir}/{video_stem}: Error - CSV read failed: {e}",
            "class_name": video_class,
            "candidates": [],
        }

    original_video_file = find_source_video(
        source_videos_root, relative_class_dir, video_stem, video_extensions
    )
    if original_video_file is None:
        return {
            "status": f"{relative_class_dir}/{video_stem}: Error - Original video not found under source class folder.",
            "class_name": video_class,
            "candidates": [],
        }

    props = read_video_properties(original_video_file)
    if props is None:
        return {
            "status": f"{relative_class_dir}/{video_stem}: Error - Could not open/read original video properties.",
            "class_name": video_class,
            "candidates": [],
        }
    fps, frame_width, frame_height, total_frames_original_video = props

    can_trim_based_on_data = len(df) >= trim_window_size
    can_trim_based_on_video = total_frames_original_video >= trim_window_size
    if not (can_trim_based_on_data and can_trim_based_on_video):
        return {
            "status": (
                f"{relative_class_dir}/{video_stem}: Skipping - Not enough frames in CSV "
                f"({len(df)}) or video ({total_frames_original_video}) for window {trim_window_size}."
            ),
            "class_name": video_class,
            "candidates": [],
        }

    all_evaluated_segments = evaluate_sliding_windows(
        df, neutral_column, trim_window_size, total_frames_original_video
    )

    if not all_evaluated_segments:
        return {
            "status": f"{relative_class_dir}/{video_stem}: No valid segments could be evaluated.",
            "class_name": video_class,
            "candidates": [],
        }

    # Lowest mean neutral score first (least neutral / most expressive).
    all_evaluated_segments.sort(key=lambda x: x["mean_score"])
    non_overlapping_candidates = select_non_overlapping_segments(
        all_evaluated_segments, num_top_segments
    )

    if not non_overlapping_candidates:
        return {
            "status": f"{relative_class_dir}/{video_stem}: No non-overlapping candidate segments found.",
            "class_name": video_class,
            "candidates": [],
        }

    best_score_this_video = non_overlapping_candidates[0]["mean_score"]
    relative_ceiling = min(
        best_score_this_video + relative_neutral_ceiling_slack_fraction * best_score_this_video,
        absolute_neutral_ceiling_backstop,
    )

    qualified_segments = [
        segment
        for segment in non_overlapping_candidates
        if segment["mean_score"] <= relative_ceiling
    ]

    video_key = f"{relative_class_dir}/{video_stem}"
    for segment in qualified_segments:
        segment["video_key"] = video_key
        segment["class_name"] = video_class
        segment["video_stem"] = video_stem
        segment["relative_class_dir"] = str(relative_class_dir)
        segment["original_video_file"] = str(original_video_file)
        segment["fps"] = fps
        segment["frame_width"] = frame_width
        segment["frame_height"] = frame_height

    status = (
        f"{video_key}: Scored. {len(qualified_segments)} candidates qualified "
        f"(relative ceiling {relative_ceiling:.4f}, best window {best_score_this_video:.4f})."
        if qualified_segments
        else (
            f"{video_key}: No segments from bottom {format_k_for_display(num_top_segments)} met relative "
            f"neutral-score ceiling {relative_ceiling:.4f} (best window: {best_score_this_video:.4f})."
        )
    )

    return {
        "status": status,
        "class_name": video_class,
        "candidates": qualified_segments,
    }


# =========================================================================
# Phase 2 (shared): write the final, balanced candidate set for one video
# =========================================================================


def write_candidates_for_video_worker(task_args_tuple):
    (
        video_key,
        candidates_for_video,
        trimmed_output_root_str,
        filename_suffix,
        trim_window_size,
        fourcc_codec,
        output_video_extension,
    ) = task_args_tuple

    trimmed_output_root = pathlib.Path(trimmed_output_root_str)
    pid = os.getpid()

    if not candidates_for_video:
        return f"{video_key}: Saved 0 segments (no surviving candidates)."

    first = candidates_for_video[0]
    original_video_file = pathlib.Path(first["original_video_file"])
    relative_class_dir = pathlib.Path(first["relative_class_dir"])
    video_stem = first["video_stem"]
    fps = first["fps"]
    frame_width = first["frame_width"]
    frame_height = first["frame_height"]

    log_prefix = f"{pid}-{video_key}"

    class_output_dir = trimmed_output_root / relative_class_dir
    class_output_dir.mkdir(parents=True, exist_ok=True)

    # Keep chronological order in the output filenames for readability.
    ordered_candidates = sorted(
        candidates_for_video, key=lambda c: c["video_start_frame"]
    )

    saved_count = save_qualified_segments(
        ordered_candidates,
        original_video_file,
        class_output_dir,
        video_stem,
        filename_suffix,
        trim_window_size,
        fps,
        frame_width,
        frame_height,
        fourcc_codec,
        output_video_extension,
        log_prefix,
    )

    summary = f"{video_key}: Saved {saved_count} segments."
    print(f"[{log_prefix}] Finished. {summary}")
    return summary


# =========================================================================
# Random-trimming worker (unchanged - single phase, no cross-class scoring
# to balance since selection is already random rather than score-based)
# =========================================================================


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

    log_prefix = f"{pid}-{relative_class_dir}/{video_stem}"

    class_output_dir = trimmed_output_root / relative_class_dir
    class_output_dir.mkdir(parents=True, exist_ok=True)

    props = read_video_properties(video_path)
    if props is None:
        return f"{relative_class_dir}/{video_stem}: Error - Could not open/read video properties."
    fps, frame_width, frame_height, total_frames = props

    if total_frames < trim_window_size:
        return (
            f"{relative_class_dir}/{video_stem}: Skipped - Video too short "
            f"({total_frames} frames) for window {trim_window_size}."
        )

    # Build every valid window as a candidate (same shape as the score-based
    # workers use), then shuffle so "priority order" is random instead of
    # score order, and run it through the same NMS-style greedy selector so
    # random picks never overlap each other either.
    max_valid_start_index = total_frames - trim_window_size
    all_candidate_segments = [
        {
            "video_start_frame": start_idx + 1,
            "video_end_frame": start_idx + trim_window_size,
        }
        for start_idx in range(max_valid_start_index + 1)
    ]
    np.random.shuffle(all_candidate_segments)

    selected_segments = select_non_overlapping_segments(
        all_candidate_segments, num_segments
    )
    # Keep chronological order in the output filenames/order.
    selected_segments.sort(key=lambda seg: seg["video_start_frame"])

    segments_saved_count = 0
    for i, segment_info in enumerate(selected_segments, start=1):
        start_frame_zero_based = segment_info["video_start_frame"] - 1
        output_name = f"{video_stem}_rnd_{i}{output_video_extension}"
        output_file_path = class_output_dir / output_name

        ok = write_segment(
            video_path,
            output_file_path,
            start_frame_zero_based,
            trim_window_size,
            fps,
            frame_width,
            frame_height,
            fourcc_codec,
            log_prefix,
        )
        if ok:
            segments_saved_count += 1

    duration = time.time() - start_time
    return (
        f"{relative_class_dir}/{video_stem}: Saved {segments_saved_count} random segments "
        f"in {duration:.2f}s."
    )


# =========================================================================
# Pass runners
# =========================================================================


def run_two_phase_pass(
    pass_label: str,
    score_worker_fn,
    tasks_args_list: list[tuple],
    trimmed_output_root: pathlib.Path,
    filename_suffix: str,
    trim_window_size: int,
    balance_mode: str,
    higher_is_better: bool,
    num_processes_to_use: int,
):
    """Shared two-phase driver used by both max_emotion and min_neutral passes.

    Phase 1: score every video in parallel (no writing).
    Between phases: group candidates by class, apply class-wide balancing.
    Phase 2: write the surviving candidates, grouped back by video, in parallel.
    """
    print(f"\n=== {pass_label} pass (balance_mode={balance_mode}) ===")

    if not tasks_args_list:
        print("No tasks to process.")
        return

    print(f"Found {len(tasks_args_list)} pre-analyzed folders to process.")
    print(f"Phase 1: scoring candidates with {num_processes_to_use} processes...")

    phase1_start = time.time()
    with mp.Pool(processes=num_processes_to_use) as pool:
        phase1_results = pool.map(score_worker_fn, tasks_args_list)
    print(f"Phase 1 finished in {time.time() - phase1_start:.2f}s.")

    error_or_skip_count = 0
    candidates_by_class = defaultdict(list)
    for result in phase1_results:
        print(f"  {result['status']}")
        if "Error" in result["status"] or "Skipping" in result["status"]:
            error_or_skip_count += 1
        if result["class_name"] is not None:
            candidates_by_class[result["class_name"]].extend(result["candidates"])

    print("\nPhase 1 candidate pool per class (before class-wide balancing):")
    for class_name in sorted(candidates_by_class):
        print(f"    {class_name}: {len(candidates_by_class[class_name])} candidates")

    balanced_candidates_by_class = apply_global_topk_balance(
        dict(candidates_by_class), balance_mode, higher_is_better
    )

    # Flatten back out and regroup by video for Phase 2 writing.
    all_surviving_candidates = [
        candidate
        for candidates in balanced_candidates_by_class.values()
        for candidate in candidates
    ]

    if not all_surviving_candidates:
        print("\nNo candidates survived balancing - nothing to write.")
        return

    candidates_by_video = group_candidates_by_video(all_surviving_candidates)

    phase2_tasks = [
        (
            video_key,
            video_candidates,
            str(trimmed_output_root),
            filename_suffix,
            trim_window_size,
            FOURCC_CODEC,
            OUTPUT_VIDEO_EXTENSION,
        )
        for video_key, video_candidates in candidates_by_video.items()
    ]

    print(
        f"\nPhase 2: writing {len(all_surviving_candidates)} surviving segments "
        f"across {len(phase2_tasks)} videos with {num_processes_to_use} processes..."
    )

    phase2_start = time.time()
    with mp.Pool(processes=num_processes_to_use) as pool:
        phase2_results = pool.map(write_candidates_for_video_worker, phase2_tasks)
    print(f"Phase 2 finished in {time.time() - phase2_start:.2f}s.")

    total_segments_saved = 0
    for result_summary in phase2_results:
        total_segments_saved += parse_saved_segment_count(result_summary)

    print("\n" + "=" * 50)
    print(f"{pass_label} Pass Summary:")
    print(f"Analysis folders scored: {len(tasks_args_list)} (errors/skips: {error_or_skip_count})")
    print(f"Total segments successfully written: {total_segments_saved}")
    print(f"Trimmed videos are in: {trimmed_output_root}")


def run_max_emotion_pass(args, trim_window_size):
    source_videos_root = pathlib.Path(args.source_dir)
    pre_analyzed_root = pathlib.Path(args.logs)
    trimmed_output_root = pathlib.Path(args.output_dir) / "max_emotion"

    if not source_videos_root.is_dir():
        print(f"Error: Source videos directory not found: {source_videos_root}")
        return
    if not pre_analyzed_root.is_dir():
        print(f"Error: Pre-analyzed data directory not found: {pre_analyzed_root}")
        return

    trimmed_output_root.mkdir(parents=True, exist_ok=True)

    analysis_folders_found = find_analysis_folders_imagenet(pre_analyzed_root)
    if not analysis_folders_found:
        print(f"No '*_analyze' folders found under {pre_analyzed_root}.")
        return

    # No K/--k_strategy involved here - every class's Phase 1 takes every
    # threshold-qualifying, non-overlapping window it can find. See the
    # UNCAPPED_K comment near the top of the file for why.
    tasks_args_list = [
        (
            str(folder_path),
            str(pre_analyzed_root),
            str(source_videos_root),
            CLASS_TO_EMOTION_MAP,
            VIDEO_EXTENSIONS,
            EMOTIONS,
            trim_window_size,
            UNCAPPED_K,
            args.relative_score_floor_fraction,
            args.absolute_noise_floor,
        )
        for folder_path in analysis_folders_found
    ]

    num_processes_to_use = max(1, mp.cpu_count() - 1)

    run_two_phase_pass(
        pass_label="Max-emotion",
        score_worker_fn=score_single_video_max_emotion_worker,
        tasks_args_list=tasks_args_list,
        trimmed_output_root=trimmed_output_root,
        filename_suffix="max_emotion",
        trim_window_size=trim_window_size,
        balance_mode=args.balance_mode,
        higher_is_better=True,
        num_processes_to_use=num_processes_to_use,
    )


def run_min_neutral_pass(args, trim_window_size):
    source_videos_root = pathlib.Path(args.source_dir)
    pre_analyzed_root = pathlib.Path(args.logs)
    trimmed_output_root = pathlib.Path(args.output_dir) / "min_neutral"

    if not source_videos_root.is_dir():
        print(f"Error: Source videos directory not found: {source_videos_root}")
        return
    if not pre_analyzed_root.is_dir():
        print(f"Error: Pre-analyzed data directory not found: {pre_analyzed_root}")
        return

    trimmed_output_root.mkdir(parents=True, exist_ok=True)

    analysis_folders_found = find_analysis_folders_imagenet(pre_analyzed_root)
    if not analysis_folders_found:
        print(f"No '*_analyze' folders found under {pre_analyzed_root}.")
        return

    # No K/--k_strategy involved here either - see run_max_emotion_pass.
    tasks_args_list = [
        (
            str(folder_path),
            str(pre_analyzed_root),
            str(source_videos_root),
            VIDEO_EXTENSIONS,
            NEUTRAL_EMOTION_COLUMN,
            trim_window_size,
            UNCAPPED_K,
            args.relative_neutral_ceiling_slack_fraction,
            args.absolute_neutral_ceiling_backstop,
        )
        for folder_path in analysis_folders_found
    ]

    num_processes_to_use = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))

    run_two_phase_pass(
        pass_label="Min-neutral",
        score_worker_fn=score_single_video_min_neutral_worker,
        tasks_args_list=tasks_args_list,
        trimmed_output_root=trimmed_output_root,
        filename_suffix="min_neutral",
        trim_window_size=trim_window_size,
        balance_mode=args.balance_mode,
        higher_is_better=False,
        num_processes_to_use=num_processes_to_use,
    )


def run_random_pass(args, trim_window_size, per_class_k: dict | None):
    source_root = pathlib.Path(args.source_dir)
    trimmed_output_root = pathlib.Path(args.output_dir) / "random"
    trimmed_output_root.mkdir(parents=True, exist_ok=True)

    if not source_root.is_dir():
        print(f"Error: Source not found: {source_root}")
        return

    print(f"\n=== Random pass (k_strategy={args.k_strategy}) ===")
    video_files = collect_videos_imagenet(source_root, VIDEO_EXTENSIONS)

    if not video_files:
        print(f"No video files found under ImageNet-style root: {source_root}")
        return

    print(f"Found {len(video_files)} videos under: {source_root}")
    print(f"Outputting to: {trimmed_output_root}")

    tasks = [
        (
            str(video_path),
            str(source_root),
            str(trimmed_output_root),
            trim_window_size,
            get_k_for_class(
                per_class_k, video_path.parent.name, NUM_RANDOM_SEGMENTS_TO_SAMPLE
            ),
            FOURCC_CODEC,
            OUTPUT_VIDEO_EXTENSION,
        )
        for video_path in video_files
    ]

    num_procs = max(1, mp.cpu_count() - 1)
    print(f"Starting pool with {num_procs} processes...")

    start_script = time.time()
    with mp.Pool(processes=num_procs) as pool:
        results = pool.map(process_single_video_random_worker, tasks)

    success_count = 0
    total_segments = 0

    print("\n" + "=" * 50)
    print("Random Pass Summary:")
    for res in results:
        print(f"  Result: {res}")
        if "Saved" in res:
            success_count += 1
            total_segments += parse_saved_segment_count(res)

    print(f"\nProcessed {success_count}/{len(video_files)} videos successfully.")
    print(f"Total random clips generated: {total_segments}")
    print(f"Random pass finished in {time.time() - start_script:.2f}s")
    print(f"Trimmed videos are in: {trimmed_output_root}")


# =========================================================================
# CLI
# =========================================================================


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Temporal distillation: max-emotion, min-neutral, and/or random trimming, "
        "with per-class K oversampling AND class-wide top-K undersampling for imbalanced "
        "DFER datasets."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["max_emotion", "min_neutral", "random", "both"],
        default="both",
        help="Which trimming strategy to run. 'both' runs all three passes "
        "(max_emotion, min_neutral, random).",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="ImageNet-style root of cropped-face videos (input to analyze_videos.py).",
    )
    parser.add_argument(
        "--logs",
        type=str,
        required=True,
        help="Root of per-video emotion CSV logs produced by analyze_videos.py. "
        "Only required for --mode max_emotion, min_neutral, or both.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory. Max-emotion clips go in <output_dir>/max_emotion, "
        "min-neutral clips go in <output_dir>/min_neutral, "
        "random clips go in <output_dir>/random.",
    )
    parser.add_argument(
        "--clip_length",
        type=int,
        required=True,
        help="Number of frames per distilled clip (e.g. 16 for ViViT, 8 for TimeSformer).",
    )
    parser.add_argument(
        "--k_strategy",
        type=str,
        choices=K_STRATEGY_CHOICES,
        default="auto_balance",
        help="How large a CANDIDATE pool (K) to gather per video, per class, before "
        "class-wide balancing. ONLY APPLIES TO THE RANDOM PASS - max_emotion and "
        "min_neutral no longer use K at all; every class there takes every "
        "threshold-qualifying, non-overlapping window it can find, then "
        "--balance_mode syncs all classes down to the smallest class's true "
        "total (see module docstring near UNCAPPED_K). 'fixed': one global K "
        "for every class. 'manual_balance': supply K per class by hand via "
        "--manual_k. 'auto_balance' (default): K is computed automatically "
        "from class video counts so total candidates per class trend toward "
        "parity.",
    )
    parser.add_argument(
        "--manual_k",
        type=str,
        default=None,
        help="Only used when --k_strategy manual_balance (which, like all of "
        "--k_strategy, only affects the random pass). Comma-separated "
        "'class=K' pairs, e.g. \"happy=5,sad=20,angry=15,disgust=40\". "
        "Classes not listed fall back to the default base K "
        "(NUM_RANDOM_SEGMENTS_TO_SAMPLE).",
    )
    parser.add_argument(
        "--balance_mode",
        type=str,
        choices=BALANCE_MODE_CHOICES,
        default=DEFAULT_BALANCE_MODE,
        help="Final class-wide balancing step for max_emotion and min_neutral "
        "passes, applied AFTER every class's Phase 1 has taken every "
        "threshold-qualifying, non-overlapping window it can find (--relative_"
        "score_floor_fraction / --relative_neutral_ceiling_slack_fraction). "
        "'global_topk' (default): target_count = size of the smallest class's "
        "candidate pool; any class with more candidates than target_count is "
        "trimmed down to its class-wide best target_count candidates by score "
        "(pooled across all videos in that class). Classes at or below "
        "target_count keep everything. 'none': keep every qualified candidate "
        "K/gating produced, with no cross-class trimming (old behaviour).",
    )
    parser.add_argument(
        "--relative_score_floor_fraction",
        type=float,
        default=RELATIVE_SCORE_FLOOR_FRACTION,
        help="Max-emotion pass: a candidate window qualifies only if its mean "
        "target-emotion score is >= this fraction of the BEST window found "
        "in that same video (default: %(default)s). Set to 0.0 to disable "
        "relative filtering entirely (only --absolute_noise_floor still applies).",
    )
    parser.add_argument(
        "--absolute_noise_floor",
        type=float,
        default=ABSOLUTE_NOISE_FLOOR,
        help="Max-emotion pass: absolute backstop (0-100 scale) below which a "
        "window is never accepted, regardless of the relative floor "
        "(default: %(default)s).",
    )
    parser.add_argument(
        "--relative_neutral_ceiling_slack_fraction",
        type=float,
        default=RELATIVE_NEUTRAL_CEILING_SLACK_FRACTION,
        help="Min-neutral pass: a candidate window qualifies only if its mean "
        "neutral score is <= the video's own lowest (best/least-neutral) "
        "window score, plus this fraction of slack (default: %(default)s).",
    )
    parser.add_argument(
        "--absolute_neutral_ceiling_backstop",
        type=float,
        default=ABSOLUTE_NEUTRAL_CEILING_BACKSTOP,
        help="Min-neutral pass: absolute backstop (0-100 scale) above which a "
        "window is never accepted, regardless of the relative ceiling "
        "(default: %(default)s).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        print("Start method already set or cannot be forced. Continuing...")

    args = get_args()
    TRIM_WINDOW_SIZE = args.clip_length

    output_root = pathlib.Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # K/--k_strategy is only meaningful for the random pass now (max_emotion
    # and min_neutral are fully uncapped - see the UNCAPPED_K comment near
    # the top of the file). Only resolve it - and only require --manual_k
    # for manual_balance - when random is actually going to run.
    per_class_k_map = None
    if args.mode in ("random", "both"):
        per_class_k_map = resolve_per_class_k(
            k_strategy=args.k_strategy,
            base_k=NUM_TOP_SEGMENTS_TO_SELECT,
            manual_k_string=args.manual_k,
            source_root=pathlib.Path(args.source_dir),
            video_extensions=VIDEO_EXTENSIONS,
        )

    overall_start = time.time()

    if args.mode in ("max_emotion", "both"):
        run_max_emotion_pass(args, TRIM_WINDOW_SIZE)

    if args.mode in ("min_neutral", "both"):
        run_min_neutral_pass(args, TRIM_WINDOW_SIZE)

    if args.mode in ("random", "both"):
        run_random_pass(args, TRIM_WINDOW_SIZE, per_class_k_map)

    print("\n" + "=" * 50)
    print(
        f"All requested passes finished in {time.time() - overall_start:.2f} seconds."
    )
    print(f"Outputs are under: {output_root}")