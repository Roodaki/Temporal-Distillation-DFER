"""
Cross-validation protocol and evaluation utilities shared by
train_vivit.py and train_timesformer.py.

Protocol
--------
* Every source video is assigned to exactly one test fold. For each fold,
  the remaining folds form the training pool, from which a group-disjoint
  validation subset is held out (for early stopping and LR scheduling).
* Folds are built either from a predefined fold file (e.g. the official
  DFEW five-fold split, see utils/video_processing/make_fold_file.py) or with
  StratifiedGroupKFold, grouped by subject (subject-independent folds, e.g.
  Emognition) or by source video.
* All clips distilled from one source video stay in a single partition
  (train, val, or test). With --group_by subject, a subject never appears in
  more than one partition.
* Training and validation use the training clips of a video
  ("<stem>_max_emotion_<k>", "<stem>_center_<k>", "<stem>_rnd_<k>").
  Each test video is represented by exactly one clip: its label-agnostic
  "<stem>_test_1" clip when the dataset contains one, otherwise its
  top-ranked clip ("_1").
* Test predictions of all folds are pooled to compute the final metrics.
  The sample standard deviation of each metric across the test folds is
  reported alongside, and confidence intervals of the pooled metrics are
  estimated with a class-stratified bootstrap (default 10,000 resamples).
"""

from __future__ import annotations

import csv
import os
import re
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm

# Clip-name suffixes written by src/video_preprocessing/trim_videos.py.
TEST_CLIP_SUFFIX = "test"
CLIP_SUFFIXES = ["max_emotion", "center", "rnd", TEST_CLIP_SUFFIX]
_CLIP_PATTERN = re.compile(
    rf"^(.*)_({'|'.join(re.escape(s) for s in CLIP_SUFFIXES)})_(\d+)$"
)

DEFAULT_SUBJECT_REGEX = r"^([^_]+)_"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def add_protocol_args(parser):
    group = parser.add_argument_group("Evaluation protocol")
    group.add_argument(
        "--n_folds",
        type=int,
        default=3,
        help="Number of cross-validation folds. Every source video is tested exactly "
        "once and test predictions are pooled across folds. Ignored when "
        "--fold_file is given. 1 = a single grouped hold-out split of --test_split.",
    )
    group.add_argument(
        "--fold_file",
        type=str,
        default=None,
        help="CSV with columns 'video,fold' assigning every source video to a "
        "predefined test fold (e.g. the official DFEW five-fold split; build it with "
        "utils/video_processing/make_fold_file.py).",
    )
    group.add_argument(
        "--group_by",
        type=str,
        default="subject",
        choices=["subject", "video"],
        help="Unit kept together when building folds and the validation split. "
        "'subject' gives subject-independent folds (subject id parsed from the "
        "video name with --subject_regex); 'video' only keeps clips of the same "
        "source video together.",
    )
    group.add_argument(
        "--subject_regex",
        type=str,
        default=DEFAULT_SUBJECT_REGEX,
        help="Regex whose first group extracts the subject id from a source-video "
        "name (default: leading token before the first underscore, e.g. '22' in "
        "'22_AMUSEMENT_...').",
    )
    group.add_argument(
        "--val_split",
        type=float,
        default=0.15,
        help="Fraction of the training pool held out (group-disjoint) for validation.",
    )
    group.add_argument(
        "--test_split",
        type=float,
        default=0.2,
        help="Held-out fraction when --n_folds 1.",
    )
    group.add_argument(
        "--split_seed",
        type=int,
        default=1337,
        help="Seed for fold assignment, independent of --seed so folds stay fixed "
        "when sweeping training seeds.",
    )
    group.add_argument(
        "--bootstrap_resamples",
        type=int,
        default=10000,
        help="Class-stratified bootstrap resamples for confidence intervals of the "
        "pooled metrics (0 disables).",
    )
    group.add_argument(
        "--ci_level",
        type=float,
        default=0.95,
        help="Confidence level of the bootstrap intervals.",
    )
    group.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
        help="Seed for the bootstrap resampling.",
    )
    return parser


# ---------------------------------------------------------------------------
# Clip / video bookkeeping
# ---------------------------------------------------------------------------
def parse_clip_name(path):
    """Return (source_video_id, mode_suffix, clip_index) for a distilled clip."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _CLIP_PATTERN.match(stem)
    if not m:
        raise ValueError(
            f"Clip '{path}' does not match '<video>_<mode>_<index>' with mode in "
            f"{CLIP_SUFFIXES}. Refusing to treat it as an ungrouped sample, since "
            f"that could leak clips of one video across partitions."
        )
    return m.group(1), m.group(2), int(m.group(3))


def _normalize_video_id(video_id):
    video_id = str(video_id).strip()
    video_id = os.path.splitext(os.path.basename(video_id))[0]
    return str(int(video_id)) if video_id.isdigit() else video_id


class ClipTable:
    """Per-clip metadata (source video, mode, index, label, split group)."""

    def __init__(self, video_files, group_by="subject", subject_regex=DEFAULT_SUBJECT_REGEX):
        parsed = [parse_clip_name(path) for path, _ in video_files]
        self.source_ids = np.array([p[0] for p in parsed])
        self.modes = np.array([p[1] for p in parsed])
        self.clip_indices = np.array([p[2] for p in parsed])
        self.labels = np.array([label for _, label in video_files])

        if group_by == "subject":
            pattern = re.compile(subject_regex)
            groups = []
            for sid in self.source_ids:
                m = pattern.search(sid)
                if not m:
                    raise ValueError(
                        f"--subject_regex '{subject_regex}' does not match video '{sid}'. "
                        f"Pass a matching --subject_regex or use --group_by video."
                    )
                groups.append(m.group(1))
            self.groups = np.array(groups)
        else:
            self.groups = self.source_ids.copy()

        # Unique source videos, in first-seen order.
        self.videos = list(OrderedDict.fromkeys(self.source_ids.tolist()))
        first_idx = {}
        for i, sid in enumerate(self.source_ids):
            first_idx.setdefault(sid, i)
        self.video_labels = np.array([self.labels[first_idx[v]] for v in self.videos])
        self.video_groups = np.array([self.groups[first_idx[v]] for v in self.videos])
        self.has_test_clips = bool(np.any(self.modes == TEST_CLIP_SUFFIX))

    def training_clip_indices(self, videos):
        videos = set(videos)
        return [
            i
            for i, (sid, mode) in enumerate(zip(self.source_ids, self.modes))
            if sid in videos and mode != TEST_CLIP_SUFFIX
        ]

    def test_clip_indices(self, videos):
        """Exactly one clip per test video (see module docstring)."""
        videos = set(videos)
        best = {}
        for i, (sid, mode, idx) in enumerate(zip(self.source_ids, self.modes, self.clip_indices)):
            if sid not in videos:
                continue
            if self.has_test_clips:
                if mode == TEST_CLIP_SUFFIX:
                    best[sid] = i
            elif mode != TEST_CLIP_SUFFIX:
                if sid not in best or idx < self.clip_indices[best[sid]]:
                    best[sid] = i
        missing = videos - set(best)
        if missing:
            print(
                f"WARNING: {len(missing)} test video(s) have no usable test clip and are "
                f"skipped (e.g. {sorted(missing)[:5]})."
            )
        return sorted(best.values())


def assert_disjoint(indices_a, indices_b, keys, label_a, label_b):
    overlap = set(keys[i] for i in indices_a) & set(keys[i] for i in indices_b)
    assert not overlap, (
        f"LEAKAGE DETECTED between {label_a} and {label_b}: {len(overlap)} shared "
        f"group(s), e.g. {sorted(overlap)[:5]}."
    )


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------
def load_fold_file(path):
    folds = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"video", "fold"} <= set(reader.fieldnames or []):
            raise ValueError(f"{path} must have 'video' and 'fold' columns.")
        for row in reader:
            folds[_normalize_video_id(row["video"])] = int(row["fold"])
    return folds


def make_outer_folds(table, n_folds, fold_file, test_split, seed):
    """Return a list of (train_pool_videos, test_videos) pairs."""
    videos = np.array(table.videos)

    if fold_file:
        assignment = load_fold_file(fold_file)
        video_fold = np.array([assignment.get(_normalize_video_id(v), -1) for v in videos])
        unassigned = videos[video_fold < 0]
        if len(unassigned):
            print(
                f"WARNING: {len(unassigned)} video(s) are not in {fold_file} and are "
                f"excluded (e.g. {unassigned[:5].tolist()})."
            )
        fold_ids = sorted(set(video_fold[video_fold >= 0].tolist()))
        return [
            (
                videos[(video_fold >= 0) & (video_fold != k)].tolist(),
                videos[video_fold == k].tolist(),
            )
            for k in fold_ids
        ]

    if n_folds <= 1:
        n_splits = max(2, round(1.0 / test_split))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        train_rel, test_rel = next(
            sgkf.split(videos, table.video_labels, groups=table.video_groups)
        )
        return [(videos[train_rel].tolist(), videos[test_rel].tolist())]

    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [
        (videos[train_rel].tolist(), videos[test_rel].tolist())
        for train_rel, test_rel in sgkf.split(
            videos, table.video_labels, groups=table.video_groups
        )
    ]


def grouped_validation_split(table, train_pool_clip_indices, val_split, seed):
    """Group-disjoint, label-stratified train/val split of the training pool."""
    pool = np.asarray(train_pool_clip_indices)
    n_splits = max(2, round(1.0 / val_split))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_rel, val_rel = next(
        sgkf.split(pool, table.labels[pool], groups=table.groups[pool])
    )
    return pool[train_rel].tolist(), pool[val_rel].tolist()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def predict(model, loader, device):
    model.eval()
    preds_list, labels_list = [], []
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for batch in tqdm(loader, desc="Test Eval"):
            if batch is None:
                continue
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(pixel_values).logits
            preds_list.append(outputs.argmax(dim=1).cpu())
            labels_list.append(labels.cpu())
    if not labels_list:
        return np.array([], dtype=int), np.array([], dtype=int)
    return torch.cat(preds_list).numpy(), torch.cat(labels_list).numpy()


def compute_metrics(labels, preds):
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, preds, average="weighted", zero_division=0),
        # UAR: unweighted mean of per-class recall. WAR: sample-weighted
        # recall, equal to accuracy for single-label classification.
        "uar": recall_score(labels, preds, average="macro", zero_division=0),
        "war": recall_score(labels, preds, average="weighted", zero_division=0),
    }


def write_test_results(labels, preds, label_map, save_dir, suffix="", title="Test"):
    """Write per-class metrics CSV + normalized confusion matrix; return (report, metrics)."""
    class_names = [k for k, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    label_ids = list(range(len(class_names)))
    metrics = compute_metrics(labels, preds)
    print(
        f"{title} -> Acc/WAR: {metrics['accuracy']:.4f} | UAR: {metrics['uar']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f} "
        f"(n={len(labels)})"
    )

    report = classification_report(
        labels,
        preds,
        labels=label_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    csv_path = os.path.join(save_dir, f"final_class_performance{suffix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class Name", "Precision", "Recall", "F1-Score", "Support"])
        for name in class_names + ["macro avg", "weighted avg"]:
            m = report[name]
            writer.writerow(
                [
                    name,
                    f"{m['precision']:.4f}",
                    f"{m['recall']:.4f}",
                    f"{m['f1-score']:.4f}",
                    m["support"],
                ]
            )
        writer.writerow([])
        for key in ("accuracy", "war", "uar", "macro_f1", "weighted_f1"):
            writer.writerow([key.upper(), "", "", f"{metrics[key]:.4f}", len(labels)])
    print(f"Saved class-wise metrics to: {csv_path}")

    cm = confusion_matrix(labels, preds, labels=label_ids, normalize="true")
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        np.nan_to_num(cm),
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(
        f"{title} Confusion Matrix (Normalized)\n"
        f"Acc: {metrics['accuracy']:.4f}, UAR: {metrics['uar']:.4f}, "
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"test_confusion_matrix{suffix}.png"))
    plt.close()

    return report, metrics


METRIC_KEYS = ("accuracy", "war", "uar", "macro_f1", "weighted_f1")


def _sample_std(values):
    """Sample standard deviation (ddof=1) across folds; 0 for a single fold."""
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def summarize_fold_results(fold_reports, fold_metrics, save_dir):
    """Mean +/- sample std across test folds (complements the pooled results).

    Writes per-class precision/recall/F1, the macro ("Unweighted Average") and
    weighted averages, and the overall metrics. Returns
    {metric: (mean, std)} for METRIC_KEYS.
    """
    class_names = sorted(
        {
            c
            for report in fold_reports
            for c in report.keys()
            if c not in ("accuracy", "macro avg", "weighted avg")
        }
    )
    csv_path = os.path.join(save_dir, "cv_class_performance_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Mean ± sample std over {len(fold_reports)} test folds"])
        writer.writerow(
            [
                "Class Name",
                "Precision (mean ± std)",
                "Recall (mean ± std)",
                "F1 (mean ± std)",
                "Mean Support",
            ]
        )
        for cls in class_names + ["macro avg", "weighted avg"]:
            vals = {
                k: [r[cls][k] for r in fold_reports if cls in r]
                for k in ("precision", "recall", "f1-score", "support")
            }
            writer.writerow(
                [cls]
                + [
                    f"{np.mean(vals[k]):.4f} ± {_sample_std(vals[k]):.4f}"
                    for k in ("precision", "recall", "f1-score")
                ]
                + [f"{np.mean(vals['support']):.1f}"]
            )

        overall = {
            key: (
                float(np.mean([m[key] for m in fold_metrics])),
                _sample_std([m[key] for m in fold_metrics]),
            )
            for key in METRIC_KEYS
        }
        writer.writerow([])
        writer.writerow(["Metric", "Mean", "Std", "Per-fold values"])
        for key in METRIC_KEYS:
            mean, std = overall[key]
            per_fold = " ".join(f"{m[key]:.4f}" for m in fold_metrics)
            writer.writerow([key.upper(), f"{mean:.4f}", f"{std:.4f}", per_fold])
    print(f"Saved per-fold performance summary to: {csv_path}")
    return overall


def _metrics_from_confusion(cm):
    """Vectorized compute_metrics() for a stack of confusion matrices (B, K, K).

    Mirrors sklearn with zero_division=0: averages run over the classes that
    occur in the true or the predicted labels.
    """
    tp = np.diagonal(cm, axis1=1, axis2=2).astype(float)
    support = cm.sum(axis=2).astype(float)
    predicted = cm.sum(axis=1).astype(float)
    present = (support > 0) | (predicted > 0)
    n = support.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        f1 = np.where(
            precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0
        )

    n_present = present.sum(axis=1)
    accuracy = tp.sum(axis=1) / n
    return {
        "accuracy": accuracy,
        "war": accuracy,
        "uar": (recall * present).sum(axis=1) / n_present,
        "macro_f1": (f1 * present).sum(axis=1) / n_present,
        "weighted_f1": (f1 * support).sum(axis=1) / n,
    }


def bootstrap_confidence_intervals(
    labels, preds, n_resamples=10000, confidence=0.95, seed=0, batch_size=250
):
    """Class-stratified percentile bootstrap CIs of the pooled metrics.

    Each resample draws, with replacement, the same number of test videos from
    every true class as the original pooled set, so per-class support is fixed.
    Returns {metric: (low, high)} for METRIC_KEYS.
    """
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    num_classes = int(max(labels.max(), preds.max())) + 1
    class_members = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    codes = labels * num_classes + preds
    rng = np.random.default_rng(seed)

    samples = {key: [] for key in METRIC_KEYS}
    for start in range(0, n_resamples, batch_size):
        b = min(batch_size, n_resamples - start)
        idx = np.concatenate(
            [members[rng.integers(0, len(members), size=(b, len(members)))]
             for members in class_members],
            axis=1,
        )
        offsets = (np.arange(b) * num_classes * num_classes)[:, None]
        cm = np.bincount(
            (codes[idx] + offsets).ravel(), minlength=b * num_classes * num_classes
        ).reshape(b, num_classes, num_classes)
        for key, values in _metrics_from_confusion(cm).items():
            samples[key].append(values)

    tail = (1.0 - confidence) / 2.0 * 100.0
    return {
        key: tuple(
            float(v) for v in np.percentile(np.concatenate(vals), [tail, 100.0 - tail])
        )
        for key, vals in samples.items()
    }


def write_final_summary(
    path, metrics, n_folds, fold_stats=None, cis=None, confidence=0.95,
    n_resamples=0, fold_summaries=None,
):
    """Pooled metrics, fold mean +/- std, and bootstrap CIs in one text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Folds: {n_folds} (pooled test predictions)\n")
        for key in METRIC_KEYS:
            line = f"Test {key.upper()}: {metrics[key]:.4f}"
            if fold_stats:
                mean, std = fold_stats[key]
                line += f" | fold mean {mean:.4f} ± {std:.4f} (sample std)"
            if cis:
                low, high = cis[key]
                line += (
                    f" | {confidence:.0%} CI [{low:.4f}, {high:.4f}] "
                    f"(class-stratified bootstrap, {n_resamples} resamples)"
                )
            f.write(line + "\n")
        if fold_summaries:
            f.write(
                f"Val Accuracy (mean over trained folds): "
                f"{np.mean([r['acc'] for r in fold_summaries]):.4f}\n"
            )
            f.write(
                f"Val F1-Score (mean over trained folds): "
                f"{np.mean([r['f1'] for r in fold_summaries]):.4f}\n"
            )
    print(f"Saved run summary to: {path}")


def report_pooled_results(
    pooled_labels, pooled_preds, fold_reports, fold_metrics, label_map, args,
    summary_name, fold_summaries=None,
):
    """Pooled metrics + confusion matrix, fold std, bootstrap CIs, summary file."""
    labels = np.concatenate(pooled_labels)
    preds = np.concatenate(pooled_preds)
    _, metrics = write_test_results(
        labels, preds, label_map, args.save_dir, suffix="", title="Pooled test"
    )

    fold_stats = None
    if len(fold_reports) > 1:
        fold_stats = summarize_fold_results(fold_reports, fold_metrics, args.save_dir)

    cis = None
    if args.bootstrap_resamples > 0:
        cis = bootstrap_confidence_intervals(
            labels, preds, n_resamples=args.bootstrap_resamples,
            confidence=args.ci_level, seed=args.bootstrap_seed,
        )
        csv_path = os.path.join(args.save_dir, "pooled_metrics_ci.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Pooled", "CI Low", "CI High", "Level", "Resamples"])
            for key in METRIC_KEYS:
                low, high = cis[key]
                writer.writerow(
                    [key.upper(), f"{metrics[key]:.4f}", f"{low:.4f}", f"{high:.4f}",
                     args.ci_level, args.bootstrap_resamples]
                )
        print(f"Saved bootstrap confidence intervals to: {csv_path}")

    write_final_summary(
        os.path.join(args.save_dir, summary_name), metrics, len(fold_reports),
        fold_stats=fold_stats, cis=cis, confidence=args.ci_level,
        n_resamples=args.bootstrap_resamples, fold_summaries=fold_summaries,
    )
    return metrics
