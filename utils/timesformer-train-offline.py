import os

os.environ["DECORD_NUM_THREADS"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import csv
import json
import random
import argparse
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.dataloader import default_collate
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification,
    get_cosine_schedule_with_warmup,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()

from sklearn.model_selection import train_test_split
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


def compute_class_weights(dataset_indices, full_dataset, num_classes):
    """
    Calculates inverse class weights to handle dataset imbalance.
    Weight = Total_Samples / (Num_Classes * Class_Count)
    Uses num_classes from args to ensure the tensor has the correct size
    even if a class is absent from this particular split.
    """
    labels = [full_dataset.video_files[i][1] for i in dataset_indices]
    counts = Counter(labels)
    n_samples = len(labels)

    weights = [1.0] * num_classes  # default weight=1 for any class not present
    for cls_id, count in counts.items():
        weights[cls_id] = n_samples / (num_classes * count)

    return torch.FloatTensor(weights)


def validate_local_model_dir(model_path):
    """Validate that the local Hugging Face model directory exists offline."""
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Offline model directory not found: {model_path}\n"
            "Download it first on an internet-connected machine, for example:\n"
            "  hf download facebook/timesformer-base-finetuned-k400 "
            "--local-dir ./timesformer-base-finetuned-k400\n"
            "Then run this script with:\n"
            "  python timesformer-train-offline.py "
            "--model_name ./timesformer-base-finetuned-k400"
        )

    required_any_weight = [
        "pytorch_model.bin",
        "model.safetensors",
        "tf_model.h5",
        "flax_model.msgpack",
    ]
    required_files = ["config.json", "preprocessor_config.json"]

    missing = [
        filename
        for filename in required_files
        if not os.path.exists(os.path.join(model_path, filename))
    ]

    has_weight = any(
        os.path.exists(os.path.join(model_path, filename))
        for filename in required_any_weight
    )

    if missing or not has_weight:
        details = []
        if missing:
            details.append(f"missing required files: {missing}")
        if not has_weight:
            details.append(
                "missing model weights: expected one of " f"{required_any_weight}"
            )
        raise FileNotFoundError(
            f"The offline model directory is incomplete: {model_path}\n"
            + "; ".join(details)
        )


def load_processor_offline(model_path):
    """Load the TimeSformer image processor without internet access."""
    validate_local_model_dir(model_path)
    return AutoImageProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )


def load_timesformer_offline(model_path, num_classes=None):
    """Load TimeSformer from a local directory only."""
    validate_local_model_dir(model_path)
    kwargs = {"local_files_only": True}
    if num_classes is not None:
        kwargs.update(
            {
                "num_labels": num_classes,
                "ignore_mismatched_sizes": True,
            }
        )
    return TimesformerForVideoClassification.from_pretrained(model_path, **kwargs)


def get_args():
    parser = argparse.ArgumentParser(description="TimeSformer 80-20 Split Training")

    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\2. Video Facial Emotion Recognition (VFER)\Dataset\DFEW_face_trimmed8",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\2. Video Facial Emotion Recognition (VFER)\Codebase\Transformer-VFER\models\timesformer\checkpoints",
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
        default="./timesformer-base-finetuned-k400",
        help=(
            "Local path to the already-downloaded TimeSformer Hugging Face model "
            "directory. Example: ./timesformer-base-finetuned-k400"
        ),
    )
    parser.add_argument("--num_classes", type=int, default=7)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--force_rescan", action="store_true")

    return parser.parse_args()


class EmotionDataset(Dataset):
    def __init__(
        self,
        root_dir,
        processor,
        expected_num_frames,
        target_image_size,
        force_rescan=False,
    ):
        self.root_dir = root_dir
        self.processor = processor
        self.expected_num_frames = expected_num_frames
        self.target_image_size = target_image_size
        self.video_files = []
        self.label_map = {}
        self.cache_path = os.path.join(root_dir, ".timesformer_dataset_cache.json")

        if os.path.exists(self.cache_path) and not force_rescan:
            print(f"Loading dataset from cache: {self.cache_path}")
            self._load_cache()
        else:
            self._scan_and_cache()

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
            json.dump({"files": self.video_files, "map": self.label_map}, f)
        print(f"Scanned {len(self.video_files)} videos. Cache saved.")

    def _load_cache(self):
        with open(self.cache_path, "r") as f:
            data = json.load(f)
            self.video_files = data["files"]
            self.label_map = data["map"]

    def _load_frames_decord(self, video_path):
        try:
            vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
            if len(vr) == 0:
                return None
            indices = np.linspace(0, len(vr) - 1, self.expected_num_frames).astype(int)
            return list(vr.get_batch(indices).asnumpy())
        except Exception:
            return None

    def _load_frames_av(self, video_path):
        frames = []
        try:
            with av.open(video_path) as container:
                total_frames = container.streams.video[0].frames
                if total_frames > 0:
                    indices = set(
                        np.linspace(
                            0, total_frames - 1, self.expected_num_frames
                        ).astype(int)
                    )
                    for i, frame in enumerate(container.decode(video=0)):
                        if i in indices:
                            frames.append(frame.to_ndarray(format="rgb24"))
                        if len(frames) >= self.expected_num_frames:
                            break
                else:
                    for frame in container.decode(video=0):
                        frames.append(frame.to_ndarray(format="rgb24"))
                    if frames:
                        idxs = np.linspace(
                            0, len(frames) - 1, self.expected_num_frames
                        ).astype(int)
                        frames = [frames[i] for i in idxs]
        except Exception:
            return None
        return frames

    def __getitem__(self, idx):
        video_path, label = self.video_files[idx]
        frames = (
            self._load_frames_decord(video_path)
            if DECORD_AVAILABLE
            else self._load_frames_av(video_path)
        )

        if frames is None or len(frames) == 0:
            return None

        if len(frames) < self.expected_num_frames:
            frames = list(frames) + [frames[-1]] * (
                self.expected_num_frames - len(frames)
            )

        inputs = self.processor(images=frames, return_tensors="pt")
        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self):
        return len(self.video_files)


def train_epoch(model, loader, optimizer, scheduler, criterion, device, args):
    model.train()
    running_loss = 0.0
    preds_all, labels_all = [], []
    skipped_batches = 0

    device_type = device.type  # "cuda" or "cpu"
    use_amp = device_type == "cuda"

    scaler = torch.amp.GradScaler(device_type, enabled=use_amp)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        running_loss += loss.item() * args.grad_accum_steps
        _, preds = torch.max(outputs.detach(), 1)
        _, preds = torch.max(outputs.detach(), 1)
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item()*args.grad_accum_steps:.4f}"})

    remainder = (len(loader) - skipped_batches) % args.grad_accum_steps
    if remainder != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    return (
        running_loss / len(loader) if len(loader) > 0 else 0,
        accuracy_score(labels_all, preds_all) if labels_all else 0,
        f1_score(labels_all, preds_all, average="weighted") if labels_all else 0,
        skipped_batches,
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


def save_test_results(model, loader, device, label_map, save_dir):
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
    print(f"HELD-OUT TEST RESULT -> Acc: {acc:.4f} | F1: {f1:.4f}")
    class_names = [k for k, v in sorted(label_map.items(), key=lambda item: item[1])]
    report_dict = classification_report(
        labels_all, preds_all, target_names=class_names, output_dict=True
    )

    csv_path = os.path.join(save_dir, "final_class_performance.csv")
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
    plt.title(f"Test Confusion Matrix (Normalized)\nAcc: {acc:.4f}, F1: {f1:.4f}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "test_confusion_matrix.png"))
    plt.close()


def run_train(args, full_dataset, train_indices_full, device):
    """Single 80-20 stratified train/val split."""
    cv_labels = [full_dataset.video_files[i][1] for i in train_indices_full]

    actual_train_idx, actual_val_idx = train_test_split(
        train_indices_full,
        test_size=args.val_split,
        stratify=cv_labels,
        random_state=args.seed,
    )

    print(f"\nSplit -> Train: {len(actual_train_idx)} | Val: {len(actual_val_idx)}")

    class_weights = compute_class_weights(
        actual_train_idx, full_dataset, args.num_classes
    ).to(device)
    print(f"Class Weights applied: {class_weights.cpu().numpy()}")

    train_sub = Subset(full_dataset, actual_train_idx)
    val_sub = Subset(full_dataset, actual_val_idx)

    train_loader = DataLoader(
        train_sub,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=filter_none_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_sub,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=filter_none_collate,
        pin_memory=True,
    )

    print("Initializing Model...")
    model = load_timesformer_offline(args.model_name, args.num_classes)

    model.gradient_checkpointing_enable()
    model.to(device)

    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps
    )
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=args.label_smoothing
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.save_dir, f"timesformer_train_log_{timestamp}.csv")
    best_val_f1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(args.save_dir, "best_model.pth")

    for epoch in range(args.epochs):
        t_loss, t_acc, t_f1, skipped = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, args
        )
        v_loss, v_acc, v_f1 = validate(model, val_loader, criterion, device)

        print(
            f"Ep {epoch+1} | T_F1: {t_f1:.4f} | V_F1: {v_f1:.4f} (Best: {best_val_f1:.4f}) | Skip: {skipped}"
        )

        log_metrics_to_csv(
            log_file,
            [
                1,  # single run, no fold number
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
            ],
        )

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            patience_counter = 0
            atomic_save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
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
    print(f"Verified Best -> Acc: {final_acc:.4f} | F1: {final_f1:.4f}")

    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return {"acc": final_acc, "f1": final_f1}, best_model_path


def main():
    args = get_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"--- TimeSformer 80-20 Split Training ---\nModel: {args.model_name}\nVal Split: {args.val_split}\nDevice: {device}"
    )
    print("Offline mode: enabled. Loading model and processor from local files only.")

    processor = load_processor_offline(args.model_name)
    temp_conf = load_timesformer_offline(args.model_name).config

    dataset = EmotionDataset(
        args.data_dir,
        processor,
        temp_conf.num_frames,
        temp_conf.image_size,
        args.force_rescan,
    )
    if len(dataset) == 0:
        raise ValueError("Dataset Empty")

    all_indices = list(range(len(dataset)))
    all_labels = [x[1] for x in dataset.video_files]

    cv_indices, test_indices = train_test_split(
        all_indices,
        test_size=args.test_split,
        stratify=all_labels,
        random_state=args.seed,
    )

    test_sub = Subset(dataset, test_indices)
    test_loader = DataLoader(
        test_sub,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=filter_none_collate,
    )

    print(f"Data Distribution: CV Pool={len(cv_indices)}, Test Set={len(test_indices)}")

    result, best_model_path = run_train(args, dataset, cv_indices, device)

    print("\n" + "=" * 40)
    print("TRAINING SUMMARY")
    print("=" * 40)
    print(f"Val Accuracy: {result['acc']:.4f}")
    print(f"Val F1-Score: {result['f1']:.4f}")

    with open(os.path.join(args.save_dir, "final_timesformer_summary.txt"), "w") as f:
        f.write(f"Val Accuracy: {result['acc']:.4f}\n")
        f.write(f"Val F1-Score: {result['f1']:.4f}\n")

    # Final Evaluation on held-out test set
    print(f"\nEvaluating Best Model on Held-out Test Set...")
    if not os.path.exists(best_model_path):
        print("WARNING: no best model checkpoint found. Skipping test evaluation.")
        return
    model = load_timesformer_offline(args.model_name, args.num_classes).to(device)
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )

    save_test_results(model, test_loader, device, dataset.label_map, args.save_dir)


if __name__ == "__main__":
    main()
