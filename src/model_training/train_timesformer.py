import os

os.environ["DECORD_NUM_THREADS"] = "1"

import csv
import json
import random
import re
import argparse
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from transformers import (
    AutoImageProcessor,
    TimesformerConfig,  # <-- config-only load, avoids loading full weights just to read config
    TimesformerForVideoClassification,
    get_constant_schedule_with_warmup,  # linear warmup -> constant; plateau scheduler takes over after
    logging as hf_logging,
)

hf_logging.set_verbosity_error()

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tqdm import tqdm
from datetime import datetime

try:
    from decord import VideoReader, cpu

    DECORD_AVAILABLE = True
except ImportError:
    print("WARNING: 'decord' library not found. Falling back to 'av' (Slower).")
    import av

    DECORD_AVAILABLE = False


def set_seed(seed=42):
    """Sets seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def atomic_save(checkpoint, save_path):
    """Atomically saves a checkpoint."""
    temp_path = save_path + ".tmp"
    torch.save(checkpoint, temp_path)
    if os.path.exists(save_path):
        os.remove(save_path)
    os.replace(temp_path, save_path)


def log_metrics_to_csv(log_path, metrics_data):
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            header = [
                "fold",
                "epoch",
                "timestamp",
                "train_loss",
                "train_acc",
                "train_f1",
                "val_loss",
                "val_acc",
                "val_f1",
                "learning_rate",
                "skipped_batches",
                "unfrozen",
            ]
            writer.writerow(header)
        if metrics_data:
            writer.writerow(metrics_data)


def filter_none_collate(batch):
    """Drop corrupt video samples from a batch."""
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# ---------------------------------------------------------------------------
# Source-video grouping (prevents train/val/test leakage across clips drawn
# from the same source video)
# ---------------------------------------------------------------------------
# The distillation preprocessing script (max_emotion / center_clips / random
# passes) can emit multiple non-overlapping clips per source video, named
# "<video_stem>_<mode>_<index>.mp4" e.g. "00123_max_emotion_1.mp4",
# "00123_max_emotion_2.mp4", "00123_center_3.mp4", "00123_rnd_2.mp4". Clips
# sharing a video_stem come from the same subject / recording session /
# emotional episode, so a label-only stratified split (the original
# behavior) can put clip 1 of a video in train and clip 2 of the *same*
# video in test -- letting the model partly "recognize" the subject/scene
# rather than generalizing, which inflates reported metrics.
# _SOURCE_ID_SUFFIXES lists every mode suffix the preprocessing script can
# produce; extend this list if new modes are added.
#
# NOTE: "min_neutral" was removed and "center" (center_clips pass) added to
# match the updated preprocessing script (min-neutral scoring was dropped in
# favor of a fixed, per-video closest-to-midpoint "center" strategy).
_SOURCE_ID_SUFFIXES = ["max_emotion", "center", "rnd"]
_SOURCE_ID_PATTERNS = [
    re.compile(rf"^(.*)_{re.escape(suffix)}_\d+$") for suffix in _SOURCE_ID_SUFFIXES
]
_SOURCE_ID_AND_INDEX_PATTERNS = [
    re.compile(rf"^(.*)_{re.escape(suffix)}_(\d+)$") for suffix in _SOURCE_ID_SUFFIXES
]


def extract_source_id_and_clip_index(filename):
    """
    Extracts both the original source-video stem AND the trailing per-mode
    clip index (the integer after the mode suffix, e.g. 2 for
    "..._max_emotion_2.mp4") from a distilled clip filename. video_stem
    itself may contain underscores, so this anchors on the known suffix
    tokens rather than naively splitting on the last N underscores (verified
    against the exact naming used by save_qualified_segments /
    process_single_video_random_worker / process_single_video_center_worker
    in the preprocessing script).

    extract_source_id() is implemented in terms of this function so the two
    can never disagree about which suffix pattern matched.

    Raises ValueError if no known suffix pattern matches, rather than
    silently falling back to treating the clip as its own singleton group --
    a silent fallback could quietly reintroduce leakage (a source video with
    an unrecognized naming convention would never be grouped with its
    siblings) without any visible signal that something is wrong. If you add
    a new preprocessing mode, add its suffix to _SOURCE_ID_SUFFIXES first.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    for pattern in _SOURCE_ID_AND_INDEX_PATTERNS:
        m = pattern.match(stem)
        if m:
            return m.group(1), int(m.group(2))
    raise ValueError(
        f"extract_source_id_and_clip_index: filename '{filename}' (stem '{stem}') does not "
        f"match any known suffix pattern {_SOURCE_ID_SUFFIXES}. Refusing to silently treat "
        f"this as an ungrouped singleton, since that could reintroduce train/val/test leakage "
        f"for this source video. Add its suffix convention to _SOURCE_ID_SUFFIXES if this is "
        f"a legitimate new preprocessing mode."
    )


def extract_source_id(filename):
    """
    Extracts the original source-video stem from a distilled clip filename by
    stripping the known trailing "_<mode>_<index>" suffix. See
    extract_source_id_and_clip_index for the full explanation; this is a
    thin wrapper that discards the clip index.
    """
    source_id, _clip_index = extract_source_id_and_clip_index(filename)
    return source_id


def build_source_groups(video_files):
    """
    Returns an array of source-video group IDs, one per entry in
    `video_files` (a list of (path, label) tuples, in the same order as
    dataset indices), for use with StratifiedGroupKFold / group-aware splits.
    """
    return np.array([extract_source_id(path) for path, _ in video_files])


def assert_no_group_leakage(indices_a, indices_b, groups, label_a="split A", label_b="split B"):
    """
    Hard, loud sanity check: raises AssertionError immediately if any source
    -video group ID appears in both `indices_a` and `indices_b`. `groups`
    must be indexable by the *global* indices contained in indices_a/indices_b
    (i.e. the full-dataset groups array, not a pre-sliced one). Call this
    after every grouped_stratified_split -- it's cheap (set intersection) and
    turns a silent leakage regression into an immediate, unmissable failure
    rather than an inflated metric discovered much later.
    """
    groups_a = set(groups[i] for i in indices_a)
    groups_b = set(groups[i] for i in indices_b)
    overlap = groups_a & groups_b
    assert not overlap, (
        f"LEAKAGE DETECTED between {label_a} and {label_b}: {len(overlap)} source-video "
        f"group(s) appear in both splits (e.g. {sorted(overlap)[:5]}...). This means clips "
        f"from the same source video ended up on both sides of the split, which will "
        f"inflate reported metrics. This should be impossible after grouped_stratified_split "
        f"-- if you see this, check that `groups` passed to assert_no_group_leakage matches "
        f"the same indexing used to build the split."
    )


def grouped_stratified_split(indices, labels, groups, test_size, random_state, n_splits=5):
    """
    Splits `indices` into two groups, stratified by `labels` and respecting
    `groups` (no group ID appears in both halves) -- replaces plain
    train_test_split wherever multiple clips can share a source video.

    Implemented via StratifiedGroupKFold: request enough folds that one fold
    is approximately `test_size` of the data, then use that single fold as
    the held-out half. This still gives a single deterministic split (not
    full k-fold CV) when called with the default n_splits=5 for a ~20% split;
    pass a different n_splits if you need a different split fraction (e.g.
    n_splits=4 for a 25% split). random_state controls which fold is treated
    as "first" via the shuffle, so it plays the same role as in
    train_test_split.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    train_relative_idx, test_relative_idx = next(
        sgkf.split(indices, labels, groups=groups)
    )
    return indices[train_relative_idx].tolist(), indices[test_relative_idx].tolist()


def filter_test_indices_top_clip_only(test_indices, video_files):
    """
    Restricts a test-index list to only the "_1" indexed clip per source
    video, independently per mode (e.g. keeps "<stem>_max_emotion_1" and
    "<stem>_center_1" and "<stem>_rnd_1" if all exist for a video, but
    drops "_max_emotion_2", "_max_emotion_3", etc.). There is no meaningful
    way to rank "_max_emotion_1" against "_rnd_1" against each other (they
    are scored on different, incomparable criteria -- see
    score_single_video_max_emotion_worker / process_single_video_center_worker /
    process_single_video_random_worker in the preprocessing script), so "top
    clip" is defined per-mode, not as a single global winner per video.

    This only ever removes clips from the test set -- it never adds a clip,
    never moves a clip between train/val/test, and never changes which
    source videos are considered test videos. Because it operates on a
    subset of an already group-disjoint split, it cannot reintroduce
    train/val/test leakage (removing elements from one side of a disjoint
    partition can't create overlap with the other side).
    """
    filtered = []
    for idx in test_indices:
        path, _label = video_files[idx]
        _source_id, clip_index = extract_source_id_and_clip_index(path)
        if clip_index == 1:
            filtered.append(idx)
    return filtered


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------
def compute_class_weights(
    dataset_indices,
    full_dataset,
    num_classes,
    scheme="effective_number",
    beta=0.9999,
    clip_min=0.5,
    clip_max=8.0,
):
    """
    Computes per-class loss weights to handle dataset imbalance. Several
    schemes are supported since raw inverse-frequency weighting blows up
    catastrophically on classes with very low counts (e.g. ~39 samples out
    of ~11.8k), which lets a handful of examples dominate the gradient and
    starve the majority classes of signal -- this was diagnosed as the main
    driver of the val-F1 collapse seen in earlier runs.

    Schemes:
      - "inverse"          : classic weight = n_samples / (num_classes * count)
                              (kept for backwards compatibility / ablation).
      - "sqrt_inverse"      : sqrt of the above -- tempers extreme ratios.
      - "clipped_inverse"   : "inverse" scheme, then clipped to [clip_min, clip_max].
      - "effective_number"  : Cui et al. 2019 "Class-Balanced Loss Based on
                              Effective Number of Samples". weight ~
                              (1 - beta) / (1 - beta^count). This is the
                              recommended default: it is a principled,
                              published, citable scheme that saturates much
                              more gently than raw inverse-frequency weighting
                              as count shrinks, rather than an ad hoc clip.

    All schemes are finally re-normalized so that weights average to 1.0
    across the classes actually present, which keeps the overall loss
    magnitude comparable across schemes/runs.
    """
    labels = [full_dataset.video_files[i][1] for i in dataset_indices]
    counts = Counter(labels)
    n_samples = len(labels)

    weights = [1.0] * num_classes  # default weight=1 for any class not present

    if scheme == "inverse":
        for cls_id, count in counts.items():
            weights[cls_id] = n_samples / (num_classes * count)

    elif scheme == "sqrt_inverse":
        for cls_id, count in counts.items():
            weights[cls_id] = np.sqrt(n_samples / (num_classes * count))

    elif scheme == "clipped_inverse":
        for cls_id, count in counts.items():
            w = n_samples / (num_classes * count)
            weights[cls_id] = float(np.clip(w, clip_min, clip_max))

    elif scheme == "effective_number":
        for cls_id, count in counts.items():
            effective_num = 1.0 - np.power(beta, count)
            weights[cls_id] = (1.0 - beta) / max(effective_num, 1e-8)

    else:
        raise ValueError(f"Unknown class weighting scheme: {scheme}")

    weights = np.array(weights, dtype=np.float64)
    present = np.array([c in counts for c in range(num_classes)])
    if present.any():
        mean_present = weights[present].mean()
        if mean_present > 0:
            weights = weights / mean_present

    return torch.FloatTensor(weights)


def build_weighted_sampler(dataset_indices, full_dataset, num_classes):
    """
    Builds a WeightedRandomSampler that oversamples rare classes (e.g. the
    ~39-video 'disgust' class) *at the batch-sampling level*, leaving the
    underlying dataset untouched on disk -- important when the dataset
    composition must stay identical to other work for literature comparison.

    Per-sample weight = 1 / class_count, so within an epoch, expected
    exposure per class is roughly equalized. Sampling is done with
    replacement (num_samples == len(dataset_indices), one epoch's worth),
    so rare-class videos get seen multiple times per epoch while common
    classes are subsampled -- augmentation (see EmotionDataset) ensures a
    given video isn't seen as the literal same tensor every time it repeats.
    """
    labels = [full_dataset.video_files[i][1] for i in dataset_indices]
    counts = Counter(labels)
    sample_weights = [1.0 / counts[label] for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Focal loss (Lin et al., 2017), optionally combined with class weights
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal loss down-weights easy/already-confident predictions and focuses
    gradient on hard, misclassified examples, rather than relying purely on
    class-frequency weighting. Combined with class-balanced weights (the
    `weight` argument), this reproduces the "class-balanced focal loss"
    setup from Cui et al. 2019, which is a reasonable, citable choice for a
    severely imbalanced class (here, ~0.3% of samples).
    """

    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * ce_loss
        return loss.mean()


def build_criterion(args, class_weights):
    if args.loss_type == "focal":
        return FocalLoss(
            gamma=args.focal_gamma,
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)


# ---------------------------------------------------------------------------
# Backbone freezing / staged unfreezing
# ---------------------------------------------------------------------------
def freeze_backbone(model, head_attr_name="classifier"):
    """
    Freezes every parameter except the classification head. Used for the
    first `args.freeze_epochs` epochs so the model adapts the head to the
    task before any backbone weights move -- this reduces the model's
    capacity to immediately start memorizing the ~25-31 training examples
    of the smallest class, since only the head is trainable at first.
    """
    head_module = getattr(model, head_attr_name, None)
    head_param_ids = {id(p) for p in head_module.parameters()} if head_module is not None else set()
    n_frozen = 0
    for p in model.parameters():
        if id(p) not in head_param_ids:
            p.requires_grad = False
            n_frozen += 1
    print(f"Froze {n_frozen} backbone parameter tensors; head remains trainable.")


def unfreeze_backbone(model):
    """Unfreezes all parameters (staged unfreezing, after `freeze_epochs`)."""
    n_unfrozen = 0
    for p in model.parameters():
        if not p.requires_grad:
            p.requires_grad = True
            n_unfrozen += 1
    print(f"Unfroze {n_unfrozen} backbone parameter tensors; full fine-tuning resumes.")


def split_params_by_head(model, head_attr_name="classifier"):
    """
    Split model parameters into backbone / head groups using module identity
    (matching against the actual head submodule's parameters) rather than
    string-matching parameter names. This is robust to internal HF naming
    changes, unlike `"classifier" in name` substring checks.

    Falls back to substring matching (with a warning) if the model has no
    attribute called `head_attr_name`.
    """
    head_module = getattr(model, head_attr_name, None)
    if head_module is not None:
        head_param_ids = {id(p) for p in head_module.parameters()}
        head_params = [p for p in model.parameters() if id(p) in head_param_ids]
        backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
        return backbone_params, head_params

    print(
        f"WARNING: model has no attribute '{head_attr_name}'; falling back to "
        f"name-substring matching for backbone/head param groups."
    )
    backbone_params = [p for n, p in model.named_parameters() if head_attr_name not in n]
    head_params = [p for n, p in model.named_parameters() if head_attr_name in n]
    return backbone_params, head_params


def load_processor(model_name):
    """Load the TimeSformer image processor (from the Hub or a local path)."""
    return AutoImageProcessor.from_pretrained(model_name)


def load_timesformer_config(model_name):
    """
    Load only the model config (no weights) from the Hub or a local directory.
    Used in main() to read num_frames / image_size without paying the cost
    of loading the full weight file that would otherwise be discarded
    immediately (previously this script loaded the whole model just for
    `.config`).
    """
    return TimesformerConfig.from_pretrained(model_name)


def load_timesformer(model_name, num_classes=None):
    """Load TimeSformer (from the Hub or a local path)."""
    kwargs = {}
    if num_classes is not None:
        kwargs.update(
            {
                "num_labels": num_classes,
                "ignore_mismatched_sizes": True,
            }
        )
    return TimesformerForVideoClassification.from_pretrained(model_name, **kwargs)


def get_args():
    parser = argparse.ArgumentParser(description="TimeSformer 80-20 Split Training")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/distilled_clips_emotion",
        help="ImageNet-style root of distilled clips to train on (8-frame clips for TimeSformer).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints/timesformer",
        help="Directory to write model checkpoints to.",
    )

    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of training pool used for validation (default: 0.2)",
    )
    parser.add_argument("--test_split", type=float, default=0.2)

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/timesformer-base-finetuned-k400",
        help=(
            "Hugging Face Hub model id (or local path) for TimeSformer. "
            "Downloaded automatically (and cached) if not already present locally."
        ),
    )
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Fraction of an assumed full run used for linear LR warmup before "
        "handing control to the plateau-based scheduler.",
    )
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience (epochs).")

    parser.add_argument(
        "--lr_patience",
        type=int,
        default=4,
        help="Epochs of no val-F1 improvement before ReduceLROnPlateau cuts the LR.",
    )
    parser.add_argument("--lr_factor", type=float, default=0.3, help="Multiplicative LR reduction factor on plateau.")
    parser.add_argument("--min_lr", type=float, default=1e-7, help="Floor for ReduceLROnPlateau.")

    parser.add_argument("--seed", type=int, default=42, help="Seed for model init / training stochasticity.")
    parser.add_argument(
        "--split_seed",
        type=int,
        default=1337,
        help="Fixed seed for the train/test carve-out, independent of --seed, "
        "so the held-out test set doesn't move when you sweep training seeds.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--force_rescan", action="store_true")

    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a last_checkpoint.pth to resume training from (model, optimizer, "
        "scheduler states, epoch, best_val_f1, patience counter, global step).",
    )
    parser.add_argument(
        "--skip_rate_warn_threshold",
        type=float,
        default=0.05,
        help="Warn if the fraction of fully-skipped (corrupt) batches in an epoch exceeds this.",
    )

    # --- Class imbalance handling -----------------------------------------
    parser.add_argument(
        "--class_weight_scheme",
        type=str,
        default="none",
        choices=["inverse", "sqrt_inverse", "clipped_inverse", "effective_number", "none"],
        help="How per-class loss weights are computed. 'effective_number' (Cui et al. "
        "2019) is recommended for severely imbalanced classes; 'inverse' reproduces "
        "the original unbounded scheme for ablation comparisons.",
    )
    parser.add_argument(
        "--class_weight_beta",
        type=float,
        default=0.99,
        help="Beta for the 'effective_number' weighting scheme (closer to 1.0 = more "
        "aggressive correction for rare classes). NOTE: the commonly-cited default of "
        "0.9999 in Cui et al. 2019 is tuned for datasets with tens of thousands of "
        "samples per class (e.g. iNaturalist); at that beta, classes in the "
        "hundreds-to-low-thousands range (like this dataset's ~39-3413 per class) "
        "barely differ from uncapped inverse-frequency weighting since beta^count "
        "saturates to ~0 for all of them. 0.99 is calibrated for this dataset's "
        "actual scale -- sweep it and check the resulting weight ratios if you "
        "change dataset size significantly.",
    )
    parser.add_argument("--class_weight_clip_min", type=float, default=0.5)
    parser.add_argument("--class_weight_clip_max", type=float, default=8.0)

    parser.add_argument(
        "--loss_type",
        type=str,
        default="ce",
        choices=["ce", "focal"],
        help="'ce' = (optionally class-weighted) cross-entropy. 'focal' = focal loss "
        "(Lin et al. 2017), optionally combined with class weights for a "
        "class-balanced focal loss setup.",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0)

    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="If set, oversample rare classes during training via a WeightedRandomSampler "
        "instead of (or alongside) loss-level class weighting. Does not modify the "
        "dataset on disk -- only which samples are drawn each epoch.",
    )

    # --- Augmentation --------------------------------------------------
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable train-time augmentation (random horizontal flip, mild color jitter, "
        "temporal frame-index jitter). Disabled by default so existing runs are "
        "reproducible; recommended when combined with --use_weighted_sampler so "
        "oversampled rare-class videos aren't seen as identical repeated tensors.",
    )
    parser.add_argument("--aug_hflip_prob", type=float, default=0.5)
    parser.add_argument(
        "--aug_color_jitter",
        type=float,
        default=0.2,
        help="Max fractional change for brightness/contrast/saturation jitter.",
    )
    parser.add_argument(
        "--aug_temporal_jitter",
        type=int,
        default=1,
        help="Max +/- frame index jitter applied to each sampled frame position "
        "(before clamping to valid range).",
    )

    # --- Staged backbone freezing --------------------------------------
    parser.add_argument(
        "--freeze_epochs",
        type=int,
        default=5,
        help="Number of initial epochs where only the classification head is trained "
        "(backbone frozen). 0 disables freezing (full fine-tuning from epoch 1, "
        "matching original behavior). Recommended: 2-5 for small/imbalanced datasets.",
    )
    parser.add_argument(
        "--dropout_override",
        type=float,
        default=None,
        help="If set, overrides hidden_dropout_prob and attention_probs_dropout_prob "
        "in the model config (where supported) as an extra regularizer against "
        "small-class memorization. Leave unset to use the pretrained model's defaults.",
    )

    # --- Cross-validation ------------------------------------------------
    parser.add_argument(
        "--n_folds",
        type=int,
        default=1,
        help="If > 1, runs stratified k-fold CV over the CV pool instead of a single "
        "80/20 split, and reports mean +/- std per-class metrics across folds. "
        "Recommended for small classes (e.g. ~39 samples) where a single split "
        "gives a high-variance, less defensible estimate.",
    )

    # --- Test-set clip thinning -------------------------------------------
    parser.add_argument(
        "--test_top_clip_only",
        action="store_true",
        help="If set, restrict the held-out test set to only the '_1' indexed clip "
        "per source video, independently per mode present (e.g. keep "
        "<stem>_max_emotion_1 and <stem>_center_1 and <stem>_rnd_1, but drop "
        "_max_emotion_2, _max_emotion_3, ...). Train/val are completely unaffected "
        "and still use all K clips per video; this does NOT change which source "
        "videos are assigned to train/val/test -- see filter_test_indices_top_clip_only. "
        "Default: off (original behavior -- all clips from test videos are used).",
    )

    return parser.parse_args()


def _dataset_fingerprint(root_dir):
    """
    Cheap fingerprint (file count + summed mtimes) of the dataset directory.
    Used to auto-invalidate the JSON scan cache when videos are added/removed/
    modified, instead of relying on the user remembering --force_rescan.
    Only does stat() calls -- no video decoding -- so it's fast even for
    thousands of files.
    """
    total_mtime = 0
    count = 0
    for emotion_name in sorted(os.listdir(root_dir)):
        d = os.path.join(root_dir, emotion_name)
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".mp4"):
                    count += 1
                    total_mtime += int(os.path.getmtime(os.path.join(d, f)))
    return {"count": count, "mtime_sum": total_mtime}


class EmotionDataset(Dataset):
    def __init__(
        self,
        root_dir,
        processor,
        expected_num_frames,
        target_image_size,
        force_rescan=False,
        augment=False,
        aug_hflip_prob=0.5,
        aug_color_jitter=0.2,
        aug_temporal_jitter=1,
    ):
        self.root_dir = root_dir
        self.processor = processor
        self.expected_num_frames = expected_num_frames
        self.target_image_size = target_image_size
        self.video_files = []
        self.label_map = {}
        self.cache_path = os.path.join(root_dir, ".timesformer_dataset_cache.json")

        # Augmentation is OFF by default (matches original behavior / keeps
        # eval-time transforms deterministic). Turn on with --augment; see
        # _augment_frames for what's applied. Only meaningful for splits used
        # as training data -- val/test loaders should be built with augment=False
        # by wrapping this same dataset via Subset (see run_train / main),
        # since Subset does not copy the underlying dataset's augment flag.
        self.augment = augment
        self.aug_hflip_prob = aug_hflip_prob
        self.aug_color_jitter = aug_color_jitter
        self.aug_temporal_jitter = aug_temporal_jitter

        if not force_rescan and self._cache_is_valid():
            print(f"Loading dataset from cache: {self.cache_path}")
            self._load_cache()
        else:
            if os.path.exists(self.cache_path) and not force_rescan:
                print(
                    "Dataset directory changed since cache was built (or cache "
                    "format is outdated) -- rescanning..."
                )
            self._scan_and_cache()

    def _cache_is_valid(self):
        if not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return False
        cached_fp = data.get("fingerprint")
        if cached_fp is None:
            return False  # old cache format predating fingerprinting
        return cached_fp == _dataset_fingerprint(self.root_dir)

    def _scan_and_cache(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"{self.root_dir} not found.")
        label_idx = 0
        for emotion_name in sorted(os.listdir(self.root_dir)):
            emotion_dir = os.path.join(self.root_dir, emotion_name)
            if os.path.isdir(emotion_dir):
                self.label_map[emotion_name] = label_idx
                current_label = label_idx
                label_idx += 1
                for f in os.listdir(emotion_dir):
                    if f.lower().endswith(".mp4"):
                        self.video_files.append(
                            (os.path.join(emotion_dir, f), current_label)
                        )
        with open(self.cache_path, "w") as f:
            json.dump(
                {
                    "files": self.video_files,
                    "map": self.label_map,
                    "fingerprint": _dataset_fingerprint(self.root_dir),
                },
                f,
            )
        print(f"Scanned {len(self.video_files)} videos. Cache saved.")

    def _load_cache(self):
        with open(self.cache_path, "r") as f:
            data = json.load(f)
            self.video_files = data["files"]
            self.label_map = data["map"]

    def _load_frames_decord(self, video_path, frame_indices):
        try:
            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            if len(vr) == 0:
                return None
            indices = np.clip(frame_indices, 0, len(vr) - 1).astype(int)
            return list(vr.get_batch(indices).asnumpy())
        except Exception:
            return None

    def _load_frames_av(self, video_path, frame_indices):
        frames = []
        try:
            with av.open(video_path) as container:
                total_frames = container.streams.video[0].frames
                if total_frames > 0:
                    indices = set(np.clip(frame_indices, 0, total_frames - 1).astype(int).tolist())
                    for i, frame in enumerate(container.decode(video=0)):
                        if i in indices:
                            frames.append(frame.to_ndarray(format="rgb24"))
                        if len(frames) >= self.expected_num_frames:
                            break
                else:
                    for frame in container.decode(video=0):
                        frames.append(frame.to_ndarray(format="rgb24"))
                    if frames:
                        idxs = np.clip(frame_indices, 0, len(frames) - 1).astype(int)
                        frames = [frames[i] for i in idxs]
        except Exception:
            return None
        return frames

    def _base_frame_indices(self, total_len_hint):
        """Evenly spaced frame indices, matching the original (non-augmented) behavior."""
        return np.linspace(0, max(total_len_hint - 1, 0), self.expected_num_frames)

    def _augment_frames(self, frames):
        """
        Lightweight train-time augmentation applied consistently across all
        frames of a clip (so e.g. a flip doesn't happen on only some frames).
        Kept intentionally simple / dependency-free (numpy only):
          - random horizontal flip
          - mild brightness/contrast/saturation jitter (same factors for
            every frame in the clip, to preserve temporal consistency)
        Temporal jitter on *which* frames get sampled is applied earlier, in
        __getitem__, via the frame index computation.
        """
        if random.random() < self.aug_hflip_prob:
            frames = [np.ascontiguousarray(f[:, ::-1, :]) for f in frames]

        if self.aug_color_jitter > 0:
            b = 1.0 + random.uniform(-self.aug_color_jitter, self.aug_color_jitter)
            c = 1.0 + random.uniform(-self.aug_color_jitter, self.aug_color_jitter)
            s = 1.0 + random.uniform(-self.aug_color_jitter, self.aug_color_jitter)

            jittered = []
            for f in frames:
                arr = f.astype(np.float32)
                # brightness
                arr = arr * b
                # contrast (around per-frame mean)
                mean = arr.mean(axis=(0, 1), keepdims=True)
                arr = (arr - mean) * c + mean
                # saturation (blend with grayscale)
                gray = arr.mean(axis=2, keepdims=True)
                arr = gray + (arr - gray) * s
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                jittered.append(arr)
            frames = jittered

        return frames

    def __getitem__(self, idx):
        video_path, label = self.video_files[idx]

        # Determine a length hint cheaply by re-using decord's reader (av path
        # falls back to internal frame counting inside _load_frames_av).
        length_hint = self.expected_num_frames
        if DECORD_AVAILABLE:
            try:
                vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
                length_hint = len(vr)
            except Exception:
                return None

        base_indices = self._base_frame_indices(length_hint)

        if self.augment and self.aug_temporal_jitter > 0:
            jitter = np.random.randint(
                -self.aug_temporal_jitter, self.aug_temporal_jitter + 1, size=base_indices.shape
            )
            frame_indices = base_indices + jitter
        else:
            frame_indices = base_indices

        frames = (
            self._load_frames_decord(video_path, frame_indices)
            if DECORD_AVAILABLE
            else self._load_frames_av(video_path, frame_indices)
        )

        if frames is None or len(frames) == 0:
            return None

        if len(frames) < self.expected_num_frames:
            frames = list(frames) + [frames[-1]] * (
                self.expected_num_frames - len(frames)
            )

        if self.augment:
            frames = self._augment_frames(frames)

        inputs = self.processor(images=frames, return_tensors="pt")
        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self):
        return len(self.video_files)


class AugmentWrapper(Dataset):
    """
    Thin wrapper so a Subset of EmotionDataset can have augmentation toggled
    independently of the underlying (shared) dataset instance -- e.g. train
    split augmented, val/test splits not, without needing three separate
    EmotionDataset instances (which would triple the on-disk scan/cache cost).
    """

    def __init__(self, subset, augment):
        self.subset = subset
        self.augment = augment

    def __getitem__(self, idx):
        base_dataset = self.subset.dataset
        prev = base_dataset.augment
        base_dataset.augment = self.augment
        try:
            return self.subset[idx]
        finally:
            base_dataset.augment = prev

    def __len__(self):
        return len(self.subset)


def train_epoch(
    model,
    loader,
    optimizer,
    warmup_scheduler,
    warmup_steps,
    global_step,
    criterion,
    device,
    args,
    scaler,
):
    """
    Runs one training epoch.

    LR schedule: while `global_step < warmup_steps`, the warmup scheduler ramps
    LR linearly from 0 -> base LR on each optimizer step. Once warmup completes,
    this function stops touching the LR at all; a ReduceLROnPlateau scheduler
    (stepped once per epoch on val F1, in run_train) takes over from there.

    NOTE: `scaler` is passed in from run_train and persists across epochs,
    instead of being recreated fresh every call -- previously this reset the
    AMP loss scale to its default every epoch, undermining its whole point
    (adapting over time) and causing avoidable skipped optimizer steps while
    it re-calibrated at the start of each epoch.
    """
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []
    skipped_batches = 0

    device_type = device.type  # "cuda" or "cpu"
    use_amp = device_type == "cuda"

    optimizer.zero_grad()
    pbar = tqdm(loader, desc="Train", leave=False)

    for step, batch in enumerate(pbar):
        if batch is None:
            skipped_batches += 1
            continue

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type, enabled=use_amp):
            outputs = model(pixel_values).logits
            loss = criterion(outputs, labels) / args.grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if global_step < warmup_steps:
                warmup_scheduler.step()
            global_step += 1

        running_loss += loss.item() * args.grad_accum_steps
        _, preds = torch.max(outputs.detach(), 1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item()*args.grad_accum_steps:.4f}"})

    remainder = (len(loader) - skipped_batches) % args.grad_accum_steps
    if remainder != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if global_step < warmup_steps:
            warmup_scheduler.step()
        global_step += 1

    return (
        running_loss / len(loader) if len(loader) > 0 else 0,
        accuracy_score(labels_all, preds_all) if labels_all else 0,
        f1_score(labels_all, preds_all, average="weighted") if labels_all else 0,
        skipped_batches,
        global_step,
    )


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, preds_all, labels_all = 0.0, [], []
    device_type = device.type
    use_amp = device_type == "cuda"

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type, enabled=use_amp):
                outputs = model(pixel_values).logits
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    if not labels_all:
        return 0.0, 0.0, 0.0

    return (
        running_loss / len(loader),
        accuracy_score(labels_all, preds_all),
        f1_score(labels_all, preds_all, average="weighted"),
    )


def save_test_results(model, loader, device, label_map, save_dir, suffix=""):
    print("\nRunning predictions on Held-out Test Set...")
    model.eval()
    preds_all, labels_all = [], []
    device_type = device.type
    use_amp = device_type == "cuda"

    with torch.no_grad():
        for batch in tqdm(loader, desc="Test Eval"):
            if batch is None:
                continue
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type, enabled=use_amp):
                outputs = model(pixel_values).logits

            _, preds = torch.max(outputs, 1)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="weighted")
    macro_f1 = f1_score(labels_all, preds_all, average="macro")
    print(f"HELD-OUT TEST RESULT -> Acc: {acc:.4f} | Weighted F1: {f1:.4f} | Macro F1: {macro_f1:.4f}")
    class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    report_dict = classification_report(
        labels_all, preds_all, target_names=class_names, output_dict=True
    )

    csv_path = os.path.join(save_dir, f"final_class_performance{suffix}.csv")
    print(f"Saving class-wise metrics to: {csv_path}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class Name", "Precision", "Recall", "F1-Score", "Support"])

        for class_name, metrics in report_dict.items():
            if class_name == "accuracy":
                writer.writerow(
                    ["TOTAL ACCURACY", "", "", f"{metrics:.4f}", len(labels_all)]
                )
                continue

            writer.writerow(
                [
                    class_name,
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1-score']:.4f}",
                    metrics["support"],
                ]
            )

    cm = confusion_matrix(labels_all, preds_all, normalize="true")

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".1%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Test Confusion Matrix (Normalized)\nAcc: {acc:.4f}, Weighted F1: {f1:.4f}, Macro F1: {macro_f1:.4f}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"test_confusion_matrix{suffix}.png"))
    plt.close()

    return report_dict, acc, f1, macro_f1


def run_train(args, full_dataset, train_indices_full, device, fold_id=1):
    """Single stratified train/val split (optionally one fold of a larger k-fold CV loop),
    grouped by source video so clips from the same original video never span
    both train and val (see grouped_stratified_split / extract_source_id)."""
    cv_labels = [full_dataset.video_files[i][1] for i in train_indices_full]
    # Full-dataset-length lookup (indexable by the global indices that
    # grouped_stratified_split returns), not positionally indexed over
    # train_indices_full -- this must match how assert_no_group_leakage
    # looks up group IDs below.
    full_groups = build_source_groups(full_dataset.video_files)
    cv_groups = full_groups[train_indices_full]

    # n_splits derived from val_split so the held-out fold approximates the
    # requested fraction (e.g. val_split=0.2 -> n_splits=5 -> ~20% held out).
    val_n_splits = max(2, round(1.0 / args.val_split))
    actual_train_idx, actual_val_idx = grouped_stratified_split(
        train_indices_full,
        cv_labels,
        cv_groups,
        test_size=args.val_split,
        random_state=args.seed,
        n_splits=val_n_splits,
    )
    assert_no_group_leakage(
        actual_train_idx, actual_val_idx, full_groups, label_a="train", label_b="val"
    )

    print(f"\n[Fold {fold_id}] Split -> Train: {len(actual_train_idx)} | Val: {len(actual_val_idx)}")

    class_weights = None
    if args.class_weight_scheme != "none":
        class_weights = compute_class_weights(
            actual_train_idx,
            full_dataset,
            args.num_classes,
            scheme=args.class_weight_scheme,
            beta=args.class_weight_beta,
            clip_min=args.class_weight_clip_min,
            clip_max=args.class_weight_clip_max,
        ).to(device)
        print(f"[Fold {fold_id}] Class weights ({args.class_weight_scheme}): {class_weights.cpu().numpy()}")

    train_sub = Subset(full_dataset, actual_train_idx)
    val_sub = Subset(full_dataset, actual_val_idx)

    # Augmentation only ever applies to the train split; val/test stay
    # deterministic for a fair, reproducible evaluation.
    train_ds = AugmentWrapper(train_sub, augment=args.augment)
    val_ds = AugmentWrapper(val_sub, augment=False)

    sampler = None
    shuffle = True
    if args.use_weighted_sampler:
        sampler = build_weighted_sampler(actual_train_idx, full_dataset, args.num_classes)
        shuffle = False  # sampler and shuffle are mutually exclusive in DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=filter_none_collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=filter_none_collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    print("Initializing Model...")
    if args.dropout_override is not None:
        config = load_timesformer_config(args.model_name)
        config.num_labels = args.num_classes
        for attr in ("hidden_dropout_prob", "attention_probs_dropout_prob"):
            if hasattr(config, attr):
                setattr(config, attr, args.dropout_override)
        model = TimesformerForVideoClassification.from_pretrained(
            args.model_name, config=config, ignore_mismatched_sizes=True
        )
    else:
        model = load_timesformer(args.model_name, args.num_classes)

    model.gradient_checkpointing_enable()
    model.to(device)

    if args.freeze_epochs > 0:
        freeze_backbone(model, head_attr_name="classifier")

    backbone_params, head_params = split_params_by_head(model, head_attr_name="classifier")
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    warmup_steps = int(args.warmup_ratio * total_steps)
    warmup_scheduler = get_constant_schedule_with_warmup(optimizer, warmup_steps)
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    criterion = build_criterion(args, class_weights)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.save_dir, f"timesformer_train_log_fold{fold_id}_{timestamp}.csv")
    best_val_f1 = 0.0
    patience_counter = 0
    global_step = 0
    start_epoch = 0
    best_model_path = os.path.join(args.save_dir, f"best_model_fold{fold_id}.pth")
    last_checkpoint_path = os.path.join(args.save_dir, f"last_checkpoint_fold{fold_id}.pth")

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        warmup_scheduler.load_state_dict(ckpt["warmup_scheduler"])
        plateau_scheduler.load_state_dict(ckpt["plateau_scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best_val_f1 = ckpt["best_val_f1"]
        patience_counter = ckpt["patience_counter"]
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"] + 1
        if start_epoch >= args.freeze_epochs:
            unfreeze_backbone(model)
        print(
            f"Resumed at epoch {start_epoch}, global_step {global_step}, "
            f"best_val_f1 {best_val_f1:.4f}"
        )

    for epoch in range(start_epoch, args.epochs):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            unfreeze_backbone(model)

        t_loss, t_acc, t_f1, skipped, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            warmup_scheduler,
            warmup_steps,
            global_step,
            criterion,
            device,
            args,
            scaler,
        )
        v_loss, v_acc, v_f1 = validate(model, val_loader, criterion, device)

        skip_rate = skipped / len(train_loader) if len(train_loader) > 0 else 0
        if skip_rate > args.skip_rate_warn_threshold:
            print(
                f"WARNING: {skip_rate:.1%} of batches were fully skipped this epoch "
                f"(corrupt/unreadable videos) -- check data integrity."
            )

        # Once warmup is complete, let val F1 drive further LR reductions.
        if global_step >= warmup_steps:
            plateau_scheduler.step(v_f1)

        is_frozen = args.freeze_epochs > 0 and epoch < args.freeze_epochs
        print(
            f"[Fold {fold_id}] Ep {epoch+1} | T_F1: {t_f1:.4f} | V_F1: {v_f1:.4f} "
            f"(Best: {best_val_f1:.4f}) | Skip: {skipped} | "
            f"{'FROZEN' if is_frozen else 'unfrozen'}"
        )

        log_metrics_to_csv(
            log_file,
            [
                fold_id,
                epoch + 1,
                datetime.now(),
                t_loss,
                t_acc,
                t_f1,
                v_loss,
                v_acc,
                v_f1,
                optimizer.param_groups[0]["lr"],
                skipped,
                not is_frozen,
            ],
        )

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            patience_counter = 0
            atomic_save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Full resumable checkpoint, saved every epoch regardless of improvement.
        atomic_save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "warmup_scheduler": warmup_scheduler.state_dict(),
                "plateau_scheduler": plateau_scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_val_f1": best_val_f1,
                "patience_counter": patience_counter,
                "global_step": global_step,
            },
            last_checkpoint_path,
        )

        if patience_counter >= args.patience:
            print("Early stopping triggered.")
            break

    if not os.path.exists(best_model_path):
        print(
            "WARNING: no checkpoint was saved (val F1 never improved). Returning zero scores."
        )
        return {"acc": 0.0, "f1": 0.0}, best_model_path

    print("Reloading best model for verification...")
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )
    _, final_acc, final_f1 = validate(model, val_loader, criterion, device)
    print(f"[Fold {fold_id}] Verified Best -> Acc: {final_acc:.4f} | F1: {final_f1:.4f}")

    del model, optimizer, warmup_scheduler, plateau_scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return {"acc": final_acc, "f1": final_f1}, best_model_path


def summarize_cv_results(fold_reports, save_dir):
    """
    Aggregates per-class metrics across folds as mean +/- std, and writes a
    CSV. This is what makes a volatile small-class number (e.g. disgust,
    n~39) defensible: a single split's F1 is a high-variance point estimate,
    but "0.15 +/- 0.09 across 5 folds" tells the reader how much to trust it.
    """
    class_names = sorted(
        {c for report in fold_reports for c in report.keys() if c not in ("accuracy", "macro avg", "weighted avg")}
    )
    rows = []
    for cls in class_names:
        precisions = [r[cls]["precision"] for r in fold_reports if cls in r]
        recalls = [r[cls]["recall"] for r in fold_reports if cls in r]
        f1s = [r[cls]["f1-score"] for r in fold_reports if cls in r]
        supports = [r[cls]["support"] for r in fold_reports if cls in r]
        rows.append(
            [
                cls,
                f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
                f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
                f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
                f"{np.mean(supports):.1f}",
            ]
        )

    csv_path = os.path.join(save_dir, "cv_class_performance_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class Name", "Precision (mean ± std)", "Recall (mean ± std)", "F1 (mean ± std)", "Mean Support"])
        writer.writerows(rows)
    print(f"\nSaved cross-fold class-performance summary to: {csv_path}")


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"--- TimeSformer Training ---\nModel: {args.model_name}\nVal Split: {args.val_split}\n"
        f"Class weight scheme: {args.class_weight_scheme}\nLoss: {args.loss_type}\n"
        f"Weighted sampler: {args.use_weighted_sampler}\nAugmentation: {args.augment}\n"
        f"Freeze epochs: {args.freeze_epochs}\nFolds: {args.n_folds}\n"
        f"Test top-clip-only: {args.test_top_clip_only}\nDevice: {device}"
    )

    processor = load_processor(args.model_name)
    temp_conf = load_timesformer_config(args.model_name)

    dataset = EmotionDataset(
        args.data_dir,
        processor,
        temp_conf.num_frames,
        temp_conf.image_size,
        force_rescan=args.force_rescan,
        aug_hflip_prob=args.aug_hflip_prob,
        aug_color_jitter=args.aug_color_jitter,
        aug_temporal_jitter=args.aug_temporal_jitter,
    )
    if len(dataset) == 0:
        raise ValueError("Dataset Empty")

    all_indices = list(range(len(dataset)))
    all_labels = [x[1] for x in dataset.video_files]
    all_groups = build_source_groups(dataset.video_files)

    n_unique_sources = len(set(all_groups))
    print(
        f"Dataset has {len(all_indices)} clips from {n_unique_sources} unique source "
        f"videos ({len(all_indices) / n_unique_sources:.1f} clips/video on average)."
    )

    # NOTE: split_seed is intentionally independent of --seed, so the held-out
    # test set stays fixed even if you sweep training seeds for variance runs.
    # Grouped by source video (see extract_source_id) so multiple clips drawn
    # from the same original video (e.g. via the max_emotion/center/random
    # distillation passes with k>1) can never span both the CV pool and the
    # test set -- doing so would let the model partly recognize the subject/
    # scene instead of generalizing, inflating reported metrics.
    test_n_splits = max(2, round(1.0 / args.test_split))
    cv_indices, test_indices = grouped_stratified_split(
        all_indices,
        all_labels,
        all_groups,
        test_size=args.test_split,
        random_state=args.split_seed,
        n_splits=test_n_splits,
    )
    assert_no_group_leakage(
        cv_indices, test_indices, all_groups, label_a="CV pool", label_b="held-out test set"
    )
    print(
        f"Verified: zero source-video overlap between CV pool ({len(set(all_groups[i] for i in cv_indices))} "
        f"unique videos) and test set ({len(set(all_groups[i] for i in test_indices))} unique videos)."
    )

    # --- Optional: thin the test set down to one clip per mode per video ---
    # This ONLY removes clips from the (already-decided) test set; it never
    # changes which source videos are in train/val/test, and cannot affect
    # the leakage guarantees above (filtering a subset of a disjoint split
    # can't create overlap). See filter_test_indices_top_clip_only.
    if args.test_top_clip_only:
        pre_filter_test_indices = test_indices
        pre_filter_test_videos = set(all_groups[i] for i in pre_filter_test_indices)

        test_indices = filter_test_indices_top_clip_only(test_indices, dataset.video_files)

        # --- Sanity checks --------------------------------------------------
        # 1) Every surviving test clip really is a "_1" clip.
        surviving_clip_indices = [
            extract_source_id_and_clip_index(dataset.video_files[i][0])[1] for i in test_indices
        ]
        assert all(ci == 1 for ci in surviving_clip_indices), (
            "test_top_clip_only: filtering left a non-index-1 clip in the test set -- "
            "this should be impossible, check filter_test_indices_top_clip_only."
        )

        # 2) No source video was dropped entirely by the filter (every mode's
        #    numbering always starts at 1 for a video that produced any
        #    qualified segments in that mode, per save_qualified_segments, so
        #    filtering to index==1 should never remove a video's only
        #    representation in the test set -- verify rather than assume).
        post_filter_test_videos = set(all_groups[i] for i in test_indices)
        dropped_videos = pre_filter_test_videos - post_filter_test_videos
        assert not dropped_videos, (
            f"test_top_clip_only: {len(dropped_videos)} test source video(s) lost ALL "
            f"representation after filtering to index-1 clips (e.g. {sorted(dropped_videos)[:5]}...). "
            f"This means one or more test videos never produced a '_1' clip in any mode, "
            f"which should not be possible given how save_qualified_segments numbers clips. "
            f"Investigate before trusting held-out test metrics."
        )
        assert post_filter_test_videos.issubset(pre_filter_test_videos), (
            "test_top_clip_only: filtering somehow introduced a source video that wasn't "
            "in the test set before filtering -- this should be impossible."
        )

        # 3) Re-verify group-disjointness against the CV pool on the filtered
        #    indices (filtering a subset of an already-disjoint split can't
        #    introduce leakage, but this is cheap and removes any doubt).
        assert_no_group_leakage(
            cv_indices, test_indices, all_groups,
            label_a="CV pool", label_b="held-out test set (top-clip-only)",
        )

        print(
            f"--test_top_clip_only: reduced test set from {len(pre_filter_test_indices)} to "
            f"{len(test_indices)} clips ({len(post_filter_test_videos)} source videos retained, "
            f"same as before filtering)."
        )

    test_sub = Subset(dataset, test_indices)
    test_ds = AugmentWrapper(test_sub, augment=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=filter_none_collate,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Data Distribution: CV Pool={len(cv_indices)}, Test Set={len(test_indices)}")

    if args.n_folds <= 1:
        result, best_model_path = run_train(args, dataset, cv_indices, device, fold_id=1)

        print("\n" + "=" * 40)
        print("TRAINING SUMMARY")
        print("=" * 40)
        print(f"Val Accuracy: {result['acc']:.4f}")
        print(f"Val F1-Score: {result['f1']:.4f}")

        with open(os.path.join(args.save_dir, "final_timesformer_summary.txt"), "w") as f:
            f.write(f"Val Accuracy: {result['acc']:.4f}\n")
            f.write(f"Val F1-Score: {result['f1']:.4f}\n")

        print(f"\nEvaluating Best Model on Held-out Test Set...")
        if not os.path.exists(best_model_path):
            print("WARNING: no best model checkpoint found. Skipping test evaluation.")
            return
        model = load_timesformer(args.model_name, args.num_classes).to(device)
        model.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True)
        )
        save_test_results(model, test_loader, device, dataset.label_map, args.save_dir)

    else:
        # Stratified, group-aware k-fold CV over the CV pool. Each fold reuses
        # the *same* held-out test set (carved out once above, grouped by
        # source video), and each fold's best model is separately evaluated
        # on it -- giving mean +/- std test metrics, which is what makes
        # small-class numbers (e.g. disgust) defensible rather than a single
        # volatile split.
        #
        # NOTE: this loop does not slice cv_indices into per-fold subsets
        # itself -- each call to run_train() receives the *full* CV pool and
        # performs its own internal grouped train/val split (see
        # grouped_stratified_split in run_train). What varies per fold here
        # is fold_args.seed, which changes which source videos land in that
        # fold's internal val split. StratifiedGroupKFold is only used to
        # determine how many folds are requested / provide a consistent,
        # reproducible seed rotation; it is intentionally not used to assign
        # disjoint fold membership across iterations.
        cv_labels = [dataset.video_files[i][1] for i in cv_indices]
        cv_groups = build_source_groups([dataset.video_files[i] for i in cv_indices])
        sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.split_seed)

        fold_reports = []
        fold_summaries = []
        cv_indices_arr = np.array(cv_indices)

        for fold_id, (_, fold_val_relative_idx) in enumerate(
            sgkf.split(cv_indices_arr, cv_labels, groups=cv_groups), start=1
        ):
            # See note above: fold_val_relative_idx is intentionally unused --
            # run_train performs its own grouped train/val split internally.
            fold_args = argparse.Namespace(**vars(args))
            fold_args.seed = args.seed + fold_id  # vary the internal train/val carve per fold

            result, best_model_path = run_train(fold_args, dataset, cv_indices, device, fold_id=fold_id)
            fold_summaries.append(result)

            if os.path.exists(best_model_path):
                model = load_timesformer(args.model_name, args.num_classes).to(device)
                model.load_state_dict(
                    torch.load(best_model_path, map_location=device, weights_only=True)
                )
                report_dict, acc, f1, macro_f1 = save_test_results(
                    model, test_loader, device, dataset.label_map, args.save_dir, suffix=f"_fold{fold_id}"
                )
                fold_reports.append(report_dict)
                del model
                torch.cuda.empty_cache()

        print("\n" + "=" * 40)
        print(f"{args.n_folds}-FOLD CV SUMMARY")
        print("=" * 40)
        val_accs = [r["acc"] for r in fold_summaries]
        val_f1s = [r["f1"] for r in fold_summaries]
        print(f"Val Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
        print(f"Val F1-Score: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}")

        with open(os.path.join(args.save_dir, "final_timesformer_summary.txt"), "w") as f:
            f.write(f"Val Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}\n")
            f.write(f"Val F1-Score: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}\n")

        if fold_reports:
            summarize_cv_results(fold_reports, args.save_dir)


if __name__ == "__main__":
    main()