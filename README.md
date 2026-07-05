# Emotion-Guided Temporal Data Distillation

Official PyTorch implementation for **"Emotion-Guided Data Distillation for Spatio-Temporal Feature Learning in Video Transformer-Based Facial Expression Recognition"**.

This repository provides a modular, end-to-end pipeline for processing raw video datasets, distilling them into high-signal expressive temporal segments using emotion-based salience scoring, and fine-tuning video transformer models for dynamic facial expression recognition (DFER).

The codebase supports preprocessing, face-region extraction, emotion-guided temporal trimming, random baseline trimming, visualization utilities, and offline fine-tuning of video transformer backbones such as **ViViT** and **TimeSformer**.

---

## 🚀 Installation & Setup

We recommend using an Anaconda environment to manage dependencies.

```bash
# Clone the repository
git clone https://github.com/Roodaki/Temporal-Distillation-DFER.git
cd Temporal-Distillation-DFER

# Create and activate environment
conda create -n dfer python=3.9
conda activate dfer

# Install PyTorch
# Adjust the CUDA version according to your system if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install pipeline dependencies
pip install -r requirements.txt
```

---

## 🗂️ Codebase Architecture

The repository holds the preprocessing, temporal distillation, training, and analysis scripts. Pretrained Hugging Face backbones (e.g. `timesformer-base-finetuned-k400`) and training checkpoints are downloaded/written locally at run time and are not tracked in Git (see `.gitignore`).

```text
├── utils/
│   ├── extract_faces_mediapipe.py     # Face detection and ROI extraction using MediaPipe
│   ├── analyze_videos.py              # Frame/video-level emotion analysis using DeepFace
│   ├── trim_videos_emotion.py         # Emotion-guided temporal distillation
│   ├── trim_videos_random.py          # Random temporal trimming baseline
│   ├── organize_videos.py             # Dataset organization utilities
│   ├── get_videos_length.py           # Video duration/statistics analysis
│   ├── draw_emotion_segmentation_figure.py # Visualization of emotion-guided segmentation
│   ├── timesformer_train_offline.py   # Offline TimeSformer fine-tuning script
│   └── vivit_train_offline.py         # Offline ViViT fine-tuning script
│
├── all.bat                            # Runs the full pipeline end-to-end (Windows)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Data Distillation Pipeline

Naturalistic facial expression datasets often contain long videos with neutral, low-signal, or redundant frames. This repository implements an emotion-guided temporal distillation pipeline that selects the most expressive temporal segments from each video before training.

The general pipeline is:

```text
Raw videos
   ↓
Face ROI extraction
   ↓
Emotion probability analysis
   ↓
Temporal salience scoring
   ↓
Emotion-guided clip extraction
   ↓
Video transformer fine-tuning
```

---

## 1. Face ROI Extraction

Use MediaPipe-based face extraction to crop facial regions and reduce background noise.

```bash
python utils/extract_faces_mediapipe.py \
    --input_dir ./data/raw_videos \
    --output_dir ./data/cropped_faces
```

Expected input structure:

```text
data/
└── raw_videos/
    ├── class_1/
    │   ├── video_001.mp4
    │   └── video_002.mp4
    ├── class_2/
    │   ├── video_003.mp4
    │   └── video_004.mp4
    └── ...
```

Expected output structure:

```text
data/
└── cropped_faces/
    ├── class_1/
    ├── class_2/
    └── ...
```

---

## 2. Emotion-Guided Frame/Video Analysis

After extracting face regions, use the emotion analysis script to generate emotion probability logs for each video.

```bash
python utils/analyze_videos.py \
    --data_dir ./data/cropped_faces \
    --output ./data/csv_logs
```

The analysis stage produces temporal emotion probability vectors of the form:

```text
P_t = <p_t,1, p_t,2, ..., p_t,c>
```

where `P_t` represents the predicted emotion distribution at time/frame `t`, and `c` is the number of emotion categories.

These emotion probabilities are used to identify the most expressive temporal windows in each video.

---

## 3. Emotion-Guided Temporal Distillation

The emotion-guided trimming script selects high-salience temporal clips based on the emotion probability logs.

For ViViT-style input with 16 frames:

```bash
python utils/trim_videos_emotion.py \
    --source_dir ./data/cropped_faces \
    --logs ./data/csv_logs \
    --output_dir ./data/distilled_clips_emotion \
    --clip_length 16
```

For TimeSformer-style input with 8 frames:

```bash
python utils/trim_videos_emotion.py \
    --source_dir ./data/cropped_faces \
    --logs ./data/csv_logs \
    --output_dir ./data/distilled_clips_emotion \
    --clip_length 8
```

`--source_dir` must point at the same cropped-face videos used as input to `analyze_videos.py` — the CSV logs alone only carry per-frame emotion scores, not the original frames.

The goal is to extract compact, emotion-rich clips that preserve the most discriminative temporal information while reducing redundant or neutral frames.

---

## 4. Random Temporal Trimming Baseline

A random trimming baseline is also provided for comparison against the emotion-guided distillation strategy.

```bash
python utils/trim_videos_random.py \
    --input_dir ./data/cropped_faces \
    --output_dir ./data/distilled_clips_random \
    --clip_length 16
```

For TimeSformer-style input:

```bash
python utils/trim_videos_random.py \
    --input_dir ./data/cropped_faces \
    --output_dir ./data/distilled_clips_random \
    --clip_length 8
```

This baseline allows direct comparison between random temporal sampling and emotion-guided temporal selection.

---

## 5. Visualization

To visualize emotion-guided temporal segmentation and selected expressive regions, use:

```bash
python utils/draw_emotion_segmentation_figure.py
```

This utility can be used to generate figures showing how emotion probability changes over time and which temporal segments are selected by the distillation pipeline. By default it scans `./data/csv_logs` (the output of step 2); pass `--root_dir` to point it elsewhere.

---

## 6. Optional Utilities

Two additional scripts are provided outside the core pipeline:

```bash
# Organize clips whose class is embedded in the filename (e.g. legacy/flat datasets)
# into per-class subfolders, based on the `_trimmed_<N>` / `_rnd_<N>` naming convention.
python utils/organize_videos.py \
    --source_dir ./data/distilled_clips_emotion \
    --output_dir ./data/distilled_clips_emotion_organized

# Report per-video duration, frame count, FPS, and resolution as a CSV.
python utils/get_videos_length.py \
    --dataset_root ./data/cropped_faces \
    --output_csv ./video_dataset_info.csv
```

---

## 🧠 Model Architectures & Training

This repository supports transformer-based video models for facial expression recognition.

Input tensors are expected to follow the shape:

```text
[Batch, Channels, Frames, Height, Width]
```

or:

```text
[B, C, T, H, W]
```

depending on the specific model implementation and data loader.

---

## Model Technical Specs

| Architecture | Attention Scheme                   | Input Tensor Shape     | Patch Size      | Pre-training                          |
| ------------ | ---------------------------------- | ---------------------- | --------------- | ------------------------------------- |
| TimeSformer  | Divided Space-Time Attention       | `[B, 3, 8, 224, 224]`  | 16 × 16         | ImageNet / Kinetics-style pretraining |
| ViViT        | Factorised Encoder                 | `[B, 3, 16, 224, 224]` | 16 × 16         | ImageNet / Kinetics-style pretraining |

---

## Offline Fine-Tuning

The repository includes offline training scripts for fine-tuning transformer backbones on distilled video clips. Both scripts run fully offline: they load a locally downloaded Hugging Face model directory rather than pulling weights from the Hub at train time.

First, download the pretrained backbone on any internet-connected machine:

```bash
hf download facebook/timesformer-base-finetuned-k400 --local-dir ./timesformer-base-finetuned-k400
hf download google/vivit-b-16x2-kinetics400 --local-dir ./vivit-b-16x2-kinetics400
```

### Train TimeSformer

```bash
python utils/timesformer_train_offline.py \
    --data_dir ./data/distilled_clips_emotion \
    --save_dir ./checkpoints/timesformer \
    --model_name ./timesformer-base-finetuned-k400
```

### Train ViViT

```bash
python utils/vivit_train_offline.py \
    --data_dir ./data/distilled_clips_emotion \
    --save_dir ./checkpoints/vivit \
    --model_name ./vivit-b-16x2-kinetics400
```

Both scripts accept additional flags (`--num_classes`, `--epochs`, `--batch_size`, `--lr_backbone`, `--lr_head`, `--val_split`, `--test_split`, and more) — run either script with `--help` for the full list.

---

## Fine-Tuning Configuration

The default fine-tuning setup (see `--help` on either training script to override):

| Setting                 | Value                              |
| ----------------------- | ----------------------------------- |
| Optimizer               | AdamW                              |
| Learning Rate (backbone)| `2e-5`                             |
| Learning Rate (head)    | `3e-4`                             |
| Weight Decay            | `0.01`                             |
| Scheduler               | Cosine schedule with warmup        |
| Warmup Ratio            | `0.1`                              |
| Loss Function           | Cross-Entropy Loss (label smoothing `0.1`) |
| Batch Size              | 16 (with gradient accumulation over 4 steps) |
| Early Stopping          | Validation F1-score based          |
| Early Stopping Patience | 5 epochs                           |

The pretrained backbone and the classification head are trained with separate learning rates (`--lr_backbone` / `--lr_head`), which is a lightweight approximation of discriminative fine-tuning.

---

## 📝 Citation

If you use this codebase or the emotion-guided temporal distillation methodology, please cite:

```bibtex
@article{roodaki2026emotion,
  title={Emotion-Guided Data Distillation for Spatio-Temporal Feature Learning in Video Transformer-Based Dynamic Facial Expression Recognition},
  author={Roodaki, AmirHossein and Sotoodeh, Mahmood and Moosavi, Mohammad R. and Mbilinyi, Ashery},
  year={2026}
}
```
