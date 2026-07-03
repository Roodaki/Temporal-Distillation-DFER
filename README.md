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
pip install deepface mediapipe opencv-python pandas numpy matplotlib scikit-learn tqdm
```

---

## 🗂️ Codebase Architecture

The repository is organized into model-related components and utility scripts for preprocessing, temporal distillation, training, and analysis.

```text
├── models/
│   ├── vivit/                         # ViViT-related model components
│   ├── timesformer/                   # TimeSformer-related model components
│   ├── videomae/                      # VideoMAE-related components/backbone utilities
│   └── checkpoints/                   # Local checkpoint directory; ignored by Git
│
├── utils/
│   ├── extract_faces_mediapipe.py     # Face detection and ROI extraction using MediaPipe
│   ├── analyze_videos.py              # Frame/video-level emotion analysis using DeepFace
│   ├── trim_videos_emotion.py         # Emotion-guided temporal distillation
│   ├── trim_videos_random.py          # Random temporal trimming baseline
│   ├── organize_videos.py             # Dataset organization utilities
│   ├── get_videos_length.py           # Video duration/statistics analysis
│   ├── draw_emotion_segmentation_figure.py # Visualization of emotion-guided segmentation
│   ├── timesformer-train-offline.py   # Offline TimeSformer fine-tuning script
│   └── vivit-train-offline.py         # Offline ViViT fine-tuning script
│
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
    --logs ./data/csv_logs \
    --output_dir ./data/distilled_clips_emotion \
    --clip_length 16
```

For TimeSformer-style input with 8 frames:

```bash
python utils/trim_videos_emotion.py \
    --logs ./data/csv_logs \
    --output_dir ./data/distilled_clips_emotion \
    --clip_length 8
```

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

This utility can be used to generate figures showing how emotion probability changes over time and which temporal segments are selected by the distillation pipeline.

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

The repository includes offline training scripts for fine-tuning transformer backbones on distilled video clips.

### Train ViViT

```bash
python utils/vivit-train-offline.py
```

### Train TimeSformer

```bash
python utils/timesformer-train-offline.py
```

Before training, make sure the dataset paths, checkpoint paths, number of classes, batch size, and training configuration inside the corresponding script match your local setup.

---

## Fine-Tuning Configuration

The standard fine-tuning setup follows:

| Setting                 | Value                     |
| ----------------------- | ------------------------- |
| Optimizer               | AdamW                     |
| Learning Rate           | `5e-5`                    |
| Weight Decay            | `0.01`                    |
| Scheduler               | ReduceLROnPlateau         |
| Scheduler Patience      | 3                         |
| Scheduler Factor        | 0.1                       |
| Minimum Learning Rate   | `1e-7`                    |
| Loss Function           | Cross-Entropy Loss        |
| Batch Size              | 8                         |
| Early Stopping          | Validation F1-score based |
| Early Stopping Patience | 15 epochs                 |
| Minimum Delta           | 0.001                     |

The pretrained backbone can be frozen while training only the final classification head, or partially unfrozen depending on the experimental setup.

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
