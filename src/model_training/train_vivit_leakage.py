from __future__ import annotations

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
    VivitImageProcessor,  # <-- use directly instead of AutoImageProcessor
    VivitConfig,  # <-- config-only load, avoids loading full weights just to read config
    VivitForVideoClassification,
    get_constant_schedule_with_warmup,  # linear warmup -> constant; plateau scheduler takes over after
    logging as hf_logging,
)

hf_logging.set_verbosity_error()

from sklearn.model_selection import train_test_split, StratifiedKFold
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
    """Sets seeds for reproducibility.

    NOTE: cudnn.benchmark is set to True for speed (cuDNN auto-tunes convolution
    algorithms for fixed input shapes). If you need bit-exact reproducibility,
    set cudnn.deterministic=True and cudnn.benchmark=False instead.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


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
    """
    Custom collate to drop 'None' samples (corrupt videos).
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


# ---------------------------------------------------------------------------
# Train/val/test splitting
# ---------------------------------------------------------------------------
# NOTE (ablation mode): this script intentionally performs a plain,
# label-stratified split over individual clips, with NO source-video
# grouping. Multiple clips distilled from the same source video (e.g.
# "00123_max_emotion_1.mp4", "00123_max_emotion_2.mp4", "00123_center_3.mp4")
# are treated as fully independent samples and may end up on *both* sides of
# a split. This is deliberate for this run (an ablation comparing against a
# grouped/leakage-safe split) -- it will inflate reported metrics relative to
# a properly grouped split, since the model can partially "recognize" a
# subject/scene it has already seen. Do not use these numbers as a
# generalization estimate; they exist only for the ablation comparison.
def stratified_split(indices, labels, test_size, random_state):
    """
    Plain label-stratified split over individual clip indices. No grouping
    by source video -- clips from the same source video may land on both
    sides of the split.
    """
    indices = np.asarray(indices)
    labels = np.asarray(labels)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return train_idx.tolist(), test_idx.tolist()


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------
def compute_class_weights(
    dataset_indices,
    full_dataset,
    num_classes,
    scheme="effective_number",
    beta=0.99,
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

    NOTE: call this BEFORE torch.compile() wraps the model -- compiled
    modules still expose the same parameters/attrs, but freezing after
    compilation is untested here and the safer order is freeze -> compile.
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
    """
    Load the ViViT image processor (from the Hub or a local path).
    Uses VivitImageProcessor directly to bypass the Auto-class registry
    lookup, which requires an 'image_processor_type' key in
    preprocessor_config.json that older model downloads may lack.
    """
    return VivitImageProcessor.from_pretrained(model_name)


def load_vivit_config(model_name):
    """
    Load only the model config (no weights) from the Hub or a local directory.
    Used in main() to read num_frames / image_size without paying the cost
    of loading the full ~300 MB weight file that would otherwise be discarded
    immediately.
    """
    return VivitConfig.from_pretrained(model_name)


def load_vivit(model_name, num_classes=None):
    """Load ViViT (from the Hub or a local path)."""
    kwargs = {}
    if num_classes is not None:
        kwargs.update(
            {
                "num_labels": num_classes,
                "ignore_mismatched_sizes": True,
            }
        )
    return VivitForVideoClassification.from_pretrained(model_name, **kwargs)


def get_args():
    parser = argparse.ArgumentParser(description="ViViT 80-20 Split Training")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/distilled_clips_emotion",
        help="ImageNet-style root of distilled clips to train on (16-frame clips for ViViT).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints/vivit",
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
        default="google/vivit-b-16x2-kinetics400",
        help=(
            "Hugging Face Hub model id (or local path) for ViViT. "
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

    # --- torch.compile ---------------------------------------------------
    parser.add_argument(
        "--use_compile",
        action="store_true",
        default=True,
        help="Apply torch.compile() to the model for faster execution (matches original "
        "script behavior, which always compiled when available). Pass --no_compile to "
        "disable -- useful when debugging, since compiled stack traces are harder to read.",
    )
    parser.add_argument("--no_compile", dest="use_compile", action="store_false")

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
        self.cache_path = os.path.join(root_dir, ".vivit_dataset_cache.json")

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
    (stepped once per epoch on val F1, in run_train) takes over from there. This
    avoids the old cosine schedule's core problem: it was sized against
    `args.epochs`, but early stopping usually cuts training short, so the LR
    never actually reached its intended minimum.
    """
    model.train()
    running_loss = 0.0
    preds_list, labels_list = [], []
    skipped_batches = 0

    device_type = device.type  # "cuda" or "cpu"
    use_amp = device_type == "cuda"

    optimizer.zero_grad(set_to_none=True)
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
            optimizer.zero_grad(set_to_none=True)
            if global_step < warmup_steps:
                warmup_scheduler.step()
            global_step += 1

        running_loss += loss.item() * args.grad_accum_steps

        _, preds = torch.max(outputs.detach(), 1)
        preds_list.append(preds.cpu())  # stay as tensor, avoid per-batch numpy()
        labels_list.append(labels.cpu())
        pbar.set_postfix({"loss": f"{loss.item()*args.grad_accum_steps:.4f}"})

    remainder = (len(loader) - skipped_batches) % args.grad_accum_steps
    if remainder != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if global_step < warmup_steps:
            warmup_scheduler.step()
        global_step += 1

    if labels_list:
        preds_all = torch.cat(preds_list).numpy()
        labels_all = torch.cat(labels_list).numpy()
    else:
        preds_all, labels_all = [], []

    return (
        running_loss / len(loader) if len(loader) > 0 else 0,
        accuracy_score(labels_all, preds_all) if len(labels_all) > 0 else 0,
        (
            f1_score(labels_all, preds_all, average="weighted")
            if len(labels_all) > 0
            else 0
        ),
        skipped_batches,
        global_step,
    )


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_list, labels_list = [], []
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
            preds_list.append(preds.cpu())  # tensor accumulation
            labels_list.append(labels.cpu())

    if not labels_list:
        return 0.0, 0.0, 0.0

    preds_all = torch.cat(preds_list).numpy()
    labels_all = torch.cat(labels_list).numpy()

    return (
        running_loss / len(loader),
        accuracy_score(labels_all, preds_all),
        f1_score(labels_all, preds_all, average="weighted"),
    )


def save_test_results(model, loader, device, label_map, save_dir, suffix=""):
    print("\nRunning predictions on Held-out Test Set...")
    model.eval()
    preds_list, labels_list = [], []
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
            preds_list.append(preds.cpu())  # tensor accumulation
            labels_list.append(labels.cpu())

    preds_all = torch.cat(preds_list).numpy()
    labels_all = torch.cat(labels_list).numpy()

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
    """Single stratified train/val split (optionally one fold of a larger k-fold CV loop).

    NOTE (ablation mode): this is a plain label-stratified split over
    individual clips -- see stratified_split(). No source-video grouping is
    applied, so clips from the same source video may appear on both sides of
    this train/val split.
    """
    cv_labels = [full_dataset.video_files[i][1] for i in train_indices_full]

    actual_train_idx, actual_val_idx = stratified_split(
        train_indices_full,
        cv_labels,
        test_size=args.val_split,
        random_state=args.seed,
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
        config = load_vivit_config(args.model_name)
        config.num_labels = args.num_classes
        for attr in ("hidden_dropout_prob", "attention_probs_dropout_prob"):
            if hasattr(config, attr):
                setattr(config, attr, args.dropout_override)
        model = VivitForVideoClassification.from_pretrained(
            args.model_name, config=config, ignore_mismatched_sizes=True
        )
    else:
        model = load_vivit(args.model_name, args.num_classes)

    model.gradient_checkpointing_enable()
    model.to(device)

    if args.freeze_epochs > 0:
        freeze_backbone(model, head_attr_name="classifier")

    # NOTE: freeze/unfreeze must operate on the *uncompiled* module, so we
    # grab backbone/head param groups before torch.compile() wraps it, and
    # freeze_backbone/unfreeze_backbone are always called on `raw_model`
    # (see below) rather than the compiled `model` object during training.
    backbone_params, head_params = split_params_by_head(model, head_attr_name="classifier")
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    if args.use_compile and hasattr(torch, "compile"):
        print("Applying torch.compile() for faster execution...")
        model = torch.compile(model)

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
    log_file = os.path.join(args.save_dir, f"vivit_train_log_fold{fold_id}_{timestamp}.csv")
    best_val_f1 = 0.0
    patience_counter = 0
    global_step = 0
    start_epoch = 0
    best_model_path = os.path.join(args.save_dir, f"best_model_fold{fold_id}.pth")
    last_checkpoint_path = os.path.join(args.save_dir, f"last_checkpoint_fold{fold_id}.pth")

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        warmup_scheduler.load_state_dict(ckpt["warmup_scheduler"])
        plateau_scheduler.load_state_dict(ckpt["plateau_scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        best_val_f1 = ckpt["best_val_f1"]
        patience_counter = ckpt["patience_counter"]
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"] + 1
        if start_epoch >= args.freeze_epochs:
            unfreeze_backbone(raw_model)
        print(
            f"Resumed at epoch {start_epoch}, global_step {global_step}, "
            f"best_val_f1 {best_val_f1:.4f}"
        )

    for epoch in range(start_epoch, args.epochs):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            unfreeze_backbone(raw_model)

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

        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            patience_counter = 0
            atomic_save(raw_model.state_dict(), best_model_path)
        else:
            patience_counter += 1

        # Full resumable checkpoint, saved every epoch regardless of improvement.
        atomic_save(
            {
                "model": raw_model.state_dict(),
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
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(
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
        f"--- ViViT Training (ABLATION: ungrouped random split) ---\n"
        f"Model: {args.model_name}\nVal Split: {args.val_split}\n"
        f"Class weight scheme: {args.class_weight_scheme}\nLoss: {args.loss_type}\n"
        f"Weighted sampler: {args.use_weighted_sampler}\nAugmentation: {args.augment}\n"
        f"Freeze epochs: {args.freeze_epochs}\ntorch.compile: {args.use_compile}\n"
        f"Folds: {args.n_folds}\nDevice: {device}"
    )

    processor = load_processor(args.model_name)
    temp_conf = load_vivit_config(args.model_name)

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

    # NOTE (ablation mode): plain label-stratified split over individual
    # clips, with NO source-video grouping -- clips from the same source
    # video can land on both sides of the CV-pool / test-set split. This is
    # intentional for this ablation run; see stratified_split() docstring.
    cv_indices, test_indices = stratified_split(
        all_indices,
        all_labels,
        test_size=args.test_split,
        random_state=args.split_seed,
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

        with open(os.path.join(args.save_dir, "final_vivit_summary.txt"), "w") as f:
            f.write(f"Val Accuracy: {result['acc']:.4f}\n")
            f.write(f"Val F1-Score: {result['f1']:.4f}\n")

        print(f"\nEvaluating Best Model on Held-out Test Set...")
        if not os.path.exists(best_model_path):
            print("WARNING: no best model checkpoint found. Skipping test evaluation.")
            return
        model = load_vivit(args.model_name, args.num_classes).to(device)
        model.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True)
        )

        save_test_results(model, test_loader, device, dataset.label_map, args.save_dir)

    else:
        # Stratified k-fold CV over the CV pool (no grouping -- ablation
        # mode). Each fold reuses the same held-out test set (carved out
        # once above), and each fold's best model is separately evaluated
        # on it -- giving mean +/- std test metrics.
        cv_labels = [dataset.video_files[i][1] for i in cv_indices]
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.split_seed)

        fold_reports = []
        fold_summaries = []
        cv_indices_arr = np.array(cv_indices)

        for fold_id, (_, fold_val_relative_idx) in enumerate(
            skf.split(cv_indices_arr, cv_labels), start=1
        ):
            # fold_val_relative_idx is intentionally unused -- run_train
            # performs its own stratified train/val split internally.
            fold_args = argparse.Namespace(**vars(args))
            fold_args.seed = args.seed + fold_id

            result, best_model_path = run_train(fold_args, dataset, cv_indices, device, fold_id=fold_id)
            fold_summaries.append(result)

            if os.path.exists(best_model_path):
                model = load_vivit(args.model_name, args.num_classes).to(device)
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

        with open(os.path.join(args.save_dir, "final_vivit_summary.txt"), "w") as f:
            f.write(f"Val Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}\n")
            f.write(f"Val F1-Score: {np.mean(val_f1s):.4f} ± {np.std(val_f1s):.4f}\n")

        if fold_reports:
            summarize_cv_results(fold_reports, args.save_dir)


if __name__ == "__main__":
    main()