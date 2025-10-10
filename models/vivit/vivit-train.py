# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os
import shutil
import av
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import VivitImageProcessor, VivitForVideoClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# ==============================================================================
# 2. CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
# --- Model & Dataset Configuration ---
MODEL_CHECKPOINT = "google/vivit-b-16x2-kinetics400"  # <--- CHANGED
NUM_EMOTION_CLASSES = 10  # Number of classes in your dataset

# --- File Paths & Directories ---
# IMPORTANT: Update these paths to your actual locations
DATASET_ROOT_DIRECTORY = "data/rotated_face224_trimmed32_organized"
# Updated save directory for the new model
SAVE_DIR = "Codebase\models\checkpoints\vivit-b-16x2-kinetics400"  # <--- CHANGED
BEST_MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "best_vivit.pth")  # <--- CHANGED
LAST_CHECKPOINT_PATH = os.path.join(SAVE_DIR, "last_vivit.pth")  # <--- CHANGED
# Persistent, disk-based log file for metrics
LOG_FILE_PATH = os.path.join(SAVE_DIR, "training_log.csv")


# --- Training Strategy ---
RESUME_TRAINING = True  # Set to True to resume from the last saved checkpoint

# --- Data Splitting ---
TEST_SET_SIZE = 0.10  # 10% of the data for the test set
VALIDATION_SET_SIZE_OF_REMAINDER = (
    0.1111  # Approx. 10% of original (0.1 / (1-0.1)) for validation
)

# --- Training Hyperparameters ---
NUM_EPOCHS = 300
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

# --- Dataloader Settings ---
# Use one less than the total number of CPU cores for stability
NUM_WORKERS = max(0, os.cpu_count())
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

# --- Learning Rate Scheduler ---
LR_SCHEDULER_PATIENCE = 3
LR_SCHEDULER_FACTOR = 0.1
LR_SCHEDULER_MIN_LR = 1e-7

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 15
MIN_DELTA = 0.001  # Minimum change in the monitored metric to qualify as an improvement


# ==============================================================================
# 3. UTILITY FUNCTIONS
# ==============================================================================
def atomic_save(checkpoint, save_path):
    """
    Atomically saves a checkpoint to avoid corruption.
    Saves to a temporary file first, then renames it to the final destination.
    """
    temp_path = save_path + ".tmp"
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, save_path)


def log_metrics_to_csv(log_path, metrics_data):
    """
    Appends a row of metrics to a CSV file.
    Creates the file and writes the header if the file does not exist.
    """
    # CORRECTED LOGIC: Check file existence before opening
    file_exists = os.path.exists(log_path)

    # Open in append mode ('a'), which creates the file if it doesn't exist
    with open(log_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)

        # Write header only if the file did not exist before opening
        if not file_exists:
            header = [
                "epoch",
                "timestamp",
                "train_loss",
                "train_acc",
                "train_f1",
                "val_loss",
                "val_acc",
                "val_f1",
                "learning_rate",
            ]
            writer.writerow(header)

        # Write data only if metrics_data is not empty
        if metrics_data:
            writer.writerow(metrics_data)


# ==============================================================================
# 4. DATASET CLASS
# ==============================================================================
class EmotionDataset(Dataset):
    """
    Custom PyTorch Dataset for loading, preprocessing, and serving video data
    for emotion recognition.
    """

    def __init__(self, root_dir, processor, expected_num_frames, target_image_size):
        self.root_dir = root_dir
        self.processor = processor
        self.expected_num_frames = expected_num_frames
        self.target_image_size = target_image_size
        self.video_files = []
        self.label_map = {}
        self.problematic_video_count = 0
        self._initialize_dataset()

    def _initialize_dataset(self):
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(
                f"The root directory {self.root_dir} was not found."
            )
        label_idx = 0
        for emotion_name in sorted(os.listdir(self.root_dir)):
            emotion_dir_path = os.path.join(self.root_dir, emotion_name)
            if os.path.isdir(emotion_dir_path):
                if emotion_name not in self.label_map:
                    self.label_map[emotion_name] = label_idx
                    label_idx += 1
                current_label = self.label_map[emotion_name]
                for video_filename in os.listdir(emotion_dir_path):
                    if video_filename.lower().endswith(".mp4"):
                        video_filepath = os.path.join(emotion_dir_path, video_filename)
                        self.video_files.append((video_filepath, current_label))
        print(f"Found {len(self.video_files)} videos in {len(self.label_map)} classes.")
        if not self.video_files:
            print(f"Warning: No .mp4 video files found in {self.root_dir}.")
        print(f"Label map created: {self.label_map}")

    def __len__(self):
        return len(self.video_files)

    def _load_video_frames(self, video_path):
        frames = []
        try:
            with av.open(video_path) as container:
                for frame in container.decode(video=0):
                    frames.append(frame.to_ndarray(format="rgb24"))
        except Exception:
            self.problematic_video_count += 1
            return []
        if not frames:
            self.problematic_video_count += 1
        return frames

    def __getitem__(self, idx):
        video_filepath, label = self.video_files[idx]
        all_frames = self._load_video_frames(video_filepath)
        if not all_frames:
            dummy_frame = np.zeros(
                (self.target_image_size, self.target_image_size, 3), dtype=np.uint8
            )
            sampled_frames = [dummy_frame] * self.expected_num_frames
        else:
            num_loaded_frames = len(all_frames)
            if num_loaded_frames < self.expected_num_frames:
                padding = [all_frames[-1]] * (
                    self.expected_num_frames - num_loaded_frames
                )
                sampled_frames = all_frames + padding
            else:
                indices = np.linspace(
                    0, num_loaded_frames - 1, self.expected_num_frames, dtype=int
                )
                sampled_frames = [all_frames[i] for i in indices]
        if len(sampled_frames) != self.expected_num_frames:
            sampled_frames = (
                sampled_frames + [sampled_frames[-1]] * self.expected_num_frames
            )[: self.expected_num_frames]
        inputs = self.processor(images=sampled_frames, return_tensors="pt")
        return {
            "pixel_values": inputs.pixel_values.squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# ==============================================================================
# 5. MAIN EXECUTION SCRIPT
# ==============================================================================
def main():
    """Main function to run the training and evaluation pipeline."""

    # --- 5.1. Setup: Model, Processor, Device ---
    print("--- 1. Initializing Model, Processor, and Device ---")
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = VivitImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    model = VivitForVideoClassification.from_pretrained(  # <--- CHANGED
        MODEL_CHECKPOINT, num_labels=NUM_EMOTION_CLASSES, ignore_mismatched_sizes=True
    )
    model.to(device)
    print(
        f"Model: {MODEL_CHECKPOINT}, Target Classes: {NUM_EMOTION_CLASSES}, Device: {device}"
    )

    # --- 5.2. Load and Prepare Full Dataset ---
    print("\n--- 2. Loading and Preparing Dataset ---")
    EXPECTED_FRAMES_FOR_MODEL = model.config.num_frames
    TARGET_IMAGE_SIZE = model.config.image_size
    print(
        f"Model expects {EXPECTED_FRAMES_FOR_MODEL} frames of size {TARGET_IMAGE_SIZE}x{TARGET_IMAGE_SIZE}"
    )

    try:
        full_dataset = EmotionDataset(
            root_dir=DATASET_ROOT_DIRECTORY,
            processor=processor,
            expected_num_frames=EXPECTED_FRAMES_FOR_MODEL,
            target_image_size=TARGET_IMAGE_SIZE,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    if len(full_dataset) == 0:
        print("Error: Dataset is empty. Exiting.")
        return
    print(f"Total problematic videos found: {full_dataset.problematic_video_count}")

    # --- 5.3. Data Splitting (Train/Validation/Test) ---
    print("\n--- 3. Splitting Data into Train, Validation, and Test Sets ---")
    all_indices = list(range(len(full_dataset)))
    all_labels = [full_dataset.video_files[i][1] for i in all_indices]
    stratify_all = all_labels if all_labels else None
    train_val_indices, test_indices = train_test_split(
        all_indices, test_size=TEST_SET_SIZE, random_state=42, stratify=stratify_all
    )
    train_val_labels = [all_labels[i] for i in train_val_indices]
    stratify_train_val = train_val_labels if train_val_labels else None
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=VALIDATION_SET_SIZE_OF_REMAINDER,
        random_state=42,
        stratify=stratify_train_val,
    )
    train_subset, val_subset, test_subset = (
        Subset(full_dataset, train_indices),
        Subset(full_dataset, val_indices),
        Subset(full_dataset, test_indices),
    )
    print(
        f"Training: {len(train_subset)}, Validation: {len(val_subset)}, Test: {len(test_subset)}"
    )

    # --- 5.4. Create DataLoaders ---
    print("\n--- 4. Creating DataLoaders ---")
    train_dataloader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f"Batch Size: {BATCH_SIZE}, Num Workers: {NUM_WORKERS}")

    # --- 5.5. Fine-Tuning Setup ---
    print("\n--- 5. Configuring Model for Fine-Tuning ---")
    for param in model.vivit.parameters():  # <--- CHANGED
        param.requires_grad = False
    print("Froze `model.vivit` backbone.")  # <--- CHANGED
    for param in model.classifier.parameters():
        param.requires_grad = True
    print("Unfroze `model.classifier` head.")
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(
        trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
    )
    print(f"Optimizer: AdamW (LR={LEARNING_RATE})")

    # --- 5.6. Resume or Initialize Training State ---
    print("\n--- 6. Loading Checkpoint or Initializing Training State ---")
    start_epoch = 0
    best_val_metric = float("-inf")
    epochs_no_improve = 0

    checkpoint_path_to_load = None
    if RESUME_TRAINING:
        if os.path.exists(LAST_CHECKPOINT_PATH):
            checkpoint_path_to_load = LAST_CHECKPOINT_PATH
            print(
                f"Found last epoch checkpoint. Resuming from: {checkpoint_path_to_load}"
            )
        elif os.path.exists(BEST_MODEL_SAVE_PATH):
            checkpoint_path_to_load = BEST_MODEL_SAVE_PATH
            print(
                f"No last epoch checkpoint. Resuming from best model: {checkpoint_path_to_load}"
            )

    if checkpoint_path_to_load:
        checkpoint = torch.load(
            checkpoint_path_to_load, map_location=device, weights_only=False
        )
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_val_metric = checkpoint.get("best_val_metric", float("-inf"))
            epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
            print(
                f"Successfully resumed from Epoch {start_epoch}. Best Val F1 was: {best_val_metric:.4f}"
            )
        except (KeyError, TypeError):
            print("Checkpoint has old format (model weights only). Loading weights.")
            model.load_state_dict(checkpoint)
    else:
        print("Starting training from scratch.")

    # --- 5.7. Training and Validation Loop ---
    # Ensure a log file exists with a header BEFORE the loop starts.
    if not os.path.exists(LOG_FILE_PATH):
        log_metrics_to_csv(LOG_FILE_PATH, [])
        print(f"Log file not found. Created a new one: {LOG_FILE_PATH}")

    print(f"\n--- 7. Starting Training Loop for {NUM_EPOCHS} Epochs ---")
    print(f"Early stopping is active with patience={EARLY_STOPPING_PATIENCE}.")

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            # --- Training Phase ---
            model.train()
            running_loss, train_preds, train_labels = 0.0, [], []
            train_pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]",
                unit="batch",
            )
            for batch in train_pbar:
                pixel_values, labels = batch["pixel_values"].to(device), batch[
                    "labels"
                ].to(device)
                optimizer.zero_grad()
                outputs = model(pixel_values).logits
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_train_loss = running_loss / len(train_dataloader)
            epoch_train_acc = accuracy_score(train_labels, train_preds)
            epoch_train_f1 = f1_score(
                train_labels, train_preds, average="weighted", zero_division=0
            )

            # --- Validation Phase ---
            model.eval()
            val_running_loss, val_preds, val_labels = 0.0, [], []
            val_pbar = tqdm(
                val_dataloader,
                desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Valid]",
                unit="batch",
            )
            with torch.no_grad():
                for batch in val_pbar:
                    pixel_values, labels = batch["pixel_values"].to(device), batch[
                        "labels"
                    ].to(device)
                    outputs = model(pixel_values).logits
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    val_pbar.set_postfix({"val_loss": f"{loss.item():.4f}"})

            epoch_val_loss = val_running_loss / len(val_dataloader)
            epoch_val_acc = accuracy_score(val_labels, val_preds)
            epoch_val_f1 = f1_score(
                val_labels, val_preds, average="weighted", zero_division=0
            )

            print(
                f"Epoch {epoch+1} Summary | Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}, F1: {epoch_train_f1:.4f} | "
                f"Valid Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.4f}, F1: {epoch_val_f1:.4f}"
            )

            # --- Scheduler, Checkpointing, Logging, and Early Stopping ---
            scheduler.step(epoch_val_loss)
            current_val_metric = epoch_val_f1

            # Get current time for timestamp
            timestamp = datetime.now().isoformat()
            current_lr = optimizer.param_groups[0]["lr"]
            metrics_row = [
                epoch + 1,
                timestamp,
                epoch_train_loss,
                epoch_train_acc,
                epoch_train_f1,
                epoch_val_loss,
                epoch_val_acc,
                epoch_val_f1,
                current_lr,
            ]
            log_metrics_to_csv(LOG_FILE_PATH, metrics_row)

            # Checkpoint the BEST model
            if current_val_metric > best_val_metric + MIN_DELTA:
                print(
                    f"Validation F1 improved ({best_val_metric:.4f} --> {current_val_metric:.4f}). Saving best model..."
                )
                best_val_metric = current_val_metric
                epochs_no_improve = 0
                atomic_save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "best_val_metric": best_val_metric,
                        "epochs_no_improve": epochs_no_improve,
                    },
                    BEST_MODEL_SAVE_PATH,
                )
            else:
                epochs_no_improve += 1
                print(
                    f"Validation F1 did not improve. No improvement epochs: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}"
                )

            # Save the LATEST state for crash recovery
            atomic_save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_metric": best_val_metric,
                    "epochs_no_improve": epochs_no_improve,
                },
                LAST_CHECKPOINT_PATH,
            )
            print(f"Latest state for epoch {epoch+1} saved to: {LAST_CHECKPOINT_PATH}")

            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(
                    f"\nEarly stopping triggered after {epoch+1} epochs. Best Val F1: {best_val_metric:.4f}"
                )
                break
    finally:
        print("\n--- Training loop finished or was interrupted. ---")

    # --- 5.8. Final Evaluation on Test Set ---
    print("\n--- 8. Evaluating on Test Set with Best Model ---")
    if os.path.exists(BEST_MODEL_SAVE_PATH):
        checkpoint = torch.load(
            BEST_MODEL_SAVE_PATH, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(
            f"Loaded best model (Val F1: {checkpoint.get('best_val_metric', 'N/A'):.4f}) for final evaluation."
        )
        model.eval()
        test_running_loss, test_preds, test_labels = 0.0, [], []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="[Test Eval]", unit="batch"):
                outputs = model(batch["pixel_values"].to(device)).logits
                loss = criterion(outputs, batch["labels"].to(device))
                test_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_preds.extend(predicted.cpu().numpy())
                test_labels.extend(batch["labels"].cpu().numpy())
        test_loss = test_running_loss / len(test_dataloader)
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average="weighted", zero_division=0)
        print(
            f"\n--- Test Set Results ---\nLoss: {test_loss:.4f}\nAccuracy: {test_acc:.4f}\nF1-Score: {test_f1:.4f}"
        )
    else:
        print("Skipping test set evaluation: No best model was saved or found.")

    # --- 5.9. Plotting Training History from CSV Log ---
    print("\n--- 9. Plotting Training History from Log File ---")
    if not os.path.exists(LOG_FILE_PATH):
        print("Log file not found. Cannot plot history.")
        return

    history = {
        "epochs": [],
        "timestamps": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": [],
    }
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                history["epochs"].append(int(row["epoch"]))
                history["timestamps"].append(row["timestamp"])
                history["train_loss"].append(float(row["train_loss"]))
                history["val_loss"].append(float(row["val_loss"]))
                history["train_acc"].append(float(row["train_acc"]))
                history["val_acc"].append(float(row["val_acc"]))
                history["train_f1"].append(float(row["train_f1"]))
                history["val_f1"].append(float(row["val_f1"]))
    except (KeyError, ValueError) as e:
        print(
            f"Error reading log file '{LOG_FILE_PATH}': {e}. It might be corrupted or have a missing header."
        )
        return

    if not history["epochs"]:
        print("No data in log file to plot.")
        return

    plt.figure(figsize=(20, 6))
    plt.subplot(1, 3, 1)
    plt.plot(history["epochs"], history["train_loss"], "bo-", label="Training Loss")
    plt.plot(history["epochs"], history["val_loss"], "ro-", label="Validation Loss")
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(history["epochs"], history["train_acc"], "bo-", label="Training Accuracy")
    plt.plot(history["epochs"], history["val_acc"], "ro-", label="Validation Accuracy")
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(history["epochs"], history["train_f1"], "bo-", label="Training F1-Score")
    plt.plot(history["epochs"], history["val_f1"], "ro-", label="Validation F1-Score")
    plt.title("F1-Score vs. Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_history.png"))
    print(f"Plot saved to: {os.path.join(SAVE_DIR, 'training_history.png')}")


if __name__ == "__main__":
    main()
