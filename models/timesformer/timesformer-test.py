# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os
import av
import torch
import numpy as np
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from tqdm import tqdm

# ==============================================================================
# 2. CONFIGURATION
# ==============================================================================
# --- Model & Dataset Configuration ---
MODEL_CHECKPOINT = "facebook/timesformer-base-finetuned-k400"
NUM_EMOTION_CLASSES = 10  # Must be the same as used in training

# --- File Paths ---
# IMPORTANT: Update these paths to your actual locations
BEST_MODEL_SAVE_PATH = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Codebase\models\timesformer\checkpoints\timesformer-base-finetuned-k400\best_timesformer.pth"
VIDEO_DIRECTORY = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Codebase\models\timesformer\test-dataset"  # <-- IMPORTANT: SET THIS TO YOUR FOLDER OF VIDEOS TO PREDICT
# This is needed to create the correct label-to-index mapping, just like in training.
DATASET_ROOT_DIRECTORY = r"C:\Users\Digi Max\Desktop\AmirHossein\University\Shiraz University\Research\Projects\Facial Emotion Recognition (FER)\Dataset\rotated_face_trimmed_organized"


# ==============================================================================
# 3. UTILITY FUNCTIONS & VIDEO PROCESSING
# ==============================================================================


def _load_video_frames(video_path):
    """
    Loads all frames from a video file using PyAV.
    """
    frames = []
    try:
        with av.open(video_path) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
    except Exception as e:
        print(f"Error loading video {os.path.basename(video_path)}: {e}")
        return []
    return frames


def _preprocess_frames(frames, processor, expected_num_frames, target_image_size):
    """
    Pads or samples frames to the expected length and applies the processor.
    Returns a tensor ready for the model.
    """
    if not frames:
        return None

    num_loaded_frames = len(frames)
    if num_loaded_frames < expected_num_frames:
        # Pad by repeating the last frame
        padding = [frames[-1]] * (expected_num_frames - num_loaded_frames)
        sampled_frames = frames + padding
    else:
        # Sample frames uniformly
        indices = np.linspace(0, num_loaded_frames - 1, expected_num_frames, dtype=int)
        sampled_frames = [frames[i] for i in indices]

    # Double-check final frame count
    if len(sampled_frames) != expected_num_frames:
        sampled_frames = (sampled_frames + [sampled_frames[-1]] * expected_num_frames)[
            :expected_num_frames
        ]

    # Process and convert to tensor
    inputs = processor(images=sampled_frames, return_tensors="pt")
    return inputs.pixel_values.squeeze(0)


# ==============================================================================
# 4. MAIN INFERENCE FUNCTION
# ==============================================================================


def predict_video_classes(model, processor, device, video_dir, label_map):
    """
    Processes all videos in a directory and predicts their emotion class.

    Args:
        model (torch.nn.Module): The loaded Timesformer model.
        processor (AutoImageProcessor): The processor for preparing frames.
        device (torch.device): The device to run inference on (e.g., 'cuda').
        video_dir (str): The path to the directory containing video files.
        label_map (dict): A dictionary mapping emotion names to integer labels.

    Returns:
        dict: A dictionary mapping video filenames to their predicted emotion.
    """
    if not os.path.exists(video_dir):
        print(f"Error: The specified video directory does not exist: {video_dir}")
        return {}

    # Create a reverse mapping from index to label name for easy interpretation
    # e.g., {0: 'happy', 1: 'sad', ...}
    idx_to_label = {v: k for k, v in label_map.items()}
    print(f"Emotion classes identified: {idx_to_label}")

    model.eval()  # Set the model to evaluation mode

    predictions = {}
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]

    if not video_files:
        print(f"No .mp4 files found in the directory: {video_dir}")
        return {}

    # Get model-specific configuration
    EXPECTED_FRAMES_FOR_MODEL = model.config.num_frames
    TARGET_IMAGE_SIZE = model.config.image_size

    print(f"\nProcessing {len(video_files)} videos...")
    with torch.no_grad():  # Disable gradient calculation for efficiency
        for video_filename in tqdm(video_files, desc="Predicting Videos", unit="video"):
            video_filepath = os.path.join(video_dir, video_filename)

            # 1. Load video frames
            raw_frames = _load_video_frames(video_filepath)
            if not raw_frames:
                predictions[video_filename] = "Error: Could not load video"
                continue

            # 2. Preprocess frames and create tensor
            pixel_values = _preprocess_frames(
                raw_frames, processor, EXPECTED_FRAMES_FOR_MODEL, TARGET_IMAGE_SIZE
            )
            pixel_values = pixel_values.to(device).unsqueeze(0)  # Add batch dimension

            # 3. Perform inference
            outputs = model(pixel_values).logits

            # 4. Get the prediction
            # Apply softmax to get probabilities and find the class with the highest probability
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)
            predicted_idx = torch.argmax(probabilities, dim=-1).item()

            # 5. Map index back to emotion label
            predicted_label = idx_to_label.get(predicted_idx, "Unknown Class")
            predictions[video_filename] = predicted_label

    return predictions


# ==============================================================================
# 5. EXECUTION SCRIPT
# ==============================================================================
if __name__ == "__main__":
    print("--- 1. Initializing Model, Processor, and Device ---")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load processor
    processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)

    # Initialize model architecture
    model = TimesformerForVideoClassification.from_pretrained(
        MODEL_CHECKPOINT, num_labels=NUM_EMOTION_CLASSES, ignore_mismatched_sizes=True
    )
    model.to(device)

    print("\n--- 2. Loading Best Model Checkpoint ---")
    if not os.path.exists(BEST_MODEL_SAVE_PATH):
        print(f"FATAL: Best model checkpoint not found at '{BEST_MODEL_SAVE_PATH}'")
        print("Please ensure the BEST_MODEL_SAVE_PATH is set correctly.")
    else:
        try:
            # Load the state dictionary from the saved checkpoint
            checkpoint = torch.load(BEST_MODEL_SAVE_PATH, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Successfully loaded best model weights.")
        except Exception as e:
            print(f"Error loading model state dictionary: {e}")
            print("The script will now exit.")
            exit()

        print("\n--- 3. Creating Label Map from Original Dataset ---")
        # This step is crucial to ensure predictions match the correct emotions.
        # It recreates the label map from the folder names in your original dataset directory.
        label_map = {}
        try:
            emotion_folders = sorted(
                [
                    d
                    for d in os.listdir(DATASET_ROOT_DIRECTORY)
                    if os.path.isdir(os.path.join(DATASET_ROOT_DIRECTORY, d))
                ]
            )
            if not emotion_folders:
                raise FileNotFoundError
            label_map = {emotion: i for i, emotion in enumerate(emotion_folders)}
        except FileNotFoundError:
            print(
                f"FATAL: Could not create label map. The original dataset directory was not found or is empty: '{DATASET_ROOT_DIRECTORY}'"
            )
            print("Please ensure DATASET_ROOT_DIRECTORY is set correctly.")
            exit()

        print("\n--- 4. Running Inference ---")
        # Get predictions
        results = predict_video_classes(
            model=model,
            processor=processor,
            device=device,
            video_dir=VIDEO_DIRECTORY,
            label_map=label_map,
        )

        print("\n--- 5. Prediction Results ---")
        if results:
            for filename, prediction in results.items():
                print(f"  - Video: '{filename}' -> Predicted Emotion: '{prediction}'")
        else:
            print("No predictions were made.")
