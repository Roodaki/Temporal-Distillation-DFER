# Emotion-Guided Temporal Data Distillation
[![DOI](https://zenodo.org/badge/1073745175.svg)](https://doi.org/10.5281/zenodo.21793814)

Official PyTorch implementation for **"Emotion-Guided Data Distillation for Spatio-Temporal Feature Learning in Video Transformer-Based Facial Expression Recognition"**.

This repository provides a modular, end-to-end pipeline for processing raw video datasets, distilling them into high-signal expressive temporal segments using emotion-based salience scoring, and fine-tuning video transformer models for dynamic facial expression recognition (DFER).

## 📖 Table of Contents
* [Installation & Setup](#-installation--setup)
* [Codebase Architecture](#️-codebase-architecture)
* [Data Distillation Pipeline](#️-data-distillation-pipeline)
  * [1. Face ROI Extraction](#1-face-roi-extraction)
  * [2. Emotion-Guided Analysis](#2-emotion-guided-framevideo-analysis)
  * [3. Temporal Distillation & Baselines](#3-temporal-distillation--baselines)
* [Utilities & Visualization](#-utilities--visualization)
* [Model Architectures & Training](#-model-architectures--training)
* [Pretrained Models](#-pretrained-models)
* [Citation](#-citation)

---

## 🚀 Installation & Setup

We recommend using an Anaconda environment to manage dependencies.

```bash
# Clone the repository
git clone https://github.com/Roodaki/Temporal-Distillation-DFER.git
cd Temporal-Distillation-DFER

# Create and activate environment
conda create -n dfer python=3.9 -y
conda activate dfer

# Install PyTorch (Adjust the CUDA version according to your system if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install pipeline dependencies
pip install -r requirements.txt
```

## 🗂️ Codebase Architecture

The repository is divided into the core pipeline (`src/`) and analysis tools (`utils/`). Pretrained Hugging Face backbones and training checkpoints are written locally at run time and are not tracked in Git.

```
.
├── src/
│   ├── model_training/           # Video transformer fine-tuning scripts
│   │   ├── train_timesformer.py
│   │   └── train_vivit.py
│   └── video_preprocessing/      # Pipeline for data extraction & distillation
│       ├── analyze_videos.py
│       ├── extract_faces_mediapipe.py
│       ├── organize_videos.py
│       ├── rotate_videos.bat
│       └── trim_videos.py        # Consolidated emotion-guided and random trimming
├── utils/
│   ├── generate_figures/         # Visualization utilities
│   │   └── draw_emotion_segmentation_figure.py
│   └── video_processing/         # Dataset statistics and analysis tools
│       ├── analyze_class_score_distributions.py
│       ├── count_clips.py
│       └── get_videos_length.py
├── requirements.txt
├── LICENSE
└── README.md
```

## ⚙️ Data Distillation Pipeline

Naturalistic facial expression datasets often contain long videos with neutral, low-signal, or redundant frames. This pipeline extracts face ROIs, computes frame-by-frame emotion salience, and distills videos into compact, high-signal clips for training.

### 1. Face ROI Extraction

Use MediaPipe to crop facial regions and reduce background noise.

```bash
python src/video_preprocessing/extract_faces_mediapipe.py \
    --input_dir ./data/raw_videos \
    --output_dir ./data/cropped_faces
```

### 2. Emotion-Guided Frame/Video Analysis

Generate emotion probability logs (`P_t = <p_t,1, ..., p_t,c>`) for each frame using DeepFace.

```bash
python src/video_preprocessing/analyze_videos.py \
    --data_dir ./data/cropped_faces \
    --output ./data/csv_logs
```

### 3. Temporal Distillation & Baselines

A single consolidated script handles emotion-guided distillation and baseline extraction (center and random clips). To handle inherently imbalanced DFER datasets, this script also features built-in class balancing strategies (`--balance_mode` and `--k_strategy`) to prevent over-represented classes from dominating the distilled dataset.

**Emotion-Guided Distillation (Max-Emotion):**
Extracts high-salience temporal clips based on emotion probability logs. Candidates are dynamically thresholded against the video's own best window to ensure weak-but-genuine expressions are preserved.

```bash
python src/video_preprocessing/trim_videos.py \
    --mode max_emotion \
    --source_dir ./data/cropped_faces \
    --logs ./data/csv_logs \
    --output_dir ./data/distilled_clips \
    --clip_length 16 \
    --balance_mode global_topk
```

**Generating Baselines (Center & Random):**
Generate baseline datasets for ablation or comparison. These do not require the emotion CSV logs.

```bash
python src/video_preprocessing/trim_videos.py \
    --mode random \
    --source_dir ./data/cropped_faces \
    --output_dir ./data/distilled_clips \
    --clip_length 16 \
    --k_strategy auto_balance
```

**Run the Full Suite:**
Passing `--mode both` extracts the emotion-guided dataset and both baselines simultaneously into three separate subdirectories (`max_emotion`, `center_clips`, and `random`).

## 📊 Utilities & Visualization

A collection of scripts is provided to manage datasets and visualize the distillation process.

```bash
# Visualize emotion-guided temporal segmentation
python utils/generate_figures/draw_emotion_segmentation_figure.py --root_dir ./data/csv_logs

# Organize flat datasets into class subfolders
python src/video_preprocessing/organize_videos.py \
    --source_dir ./data/distilled_clips/max_emotion \
    --output_dir ./data/distilled_organized

# Dataset analysis tools
python utils/video_processing/get_videos_length.py --dataset_root ./data/cropped_faces
python utils/video_processing/count_clips.py --dataset_root ./data/distilled_clips
python utils/video_processing/analyze_class_score_distributions.py --logs ./data/csv_logs
```

## 🧠 Model Architectures & Training

This repository supports transformer-based video models for facial expression recognition, specifically TimeSformer and ViViT.

The training pipeline has been engineered to handle the unique challenges of naturalistic DFER datasets, specifically severe class imbalance and train/test data leakage.

### Key Training Features

- **Strict Anti-Leakage Splitting:** Automatically extracts the original video ID from distilled clips and uses `StratifiedGroupKFold` to guarantee that clips from the same source video never cross train/val/test boundaries.
- **Advanced Imbalance Handling:** Supports raw inverse weighting, Focal Loss (`--loss_type focal`), batch-level oversampling (`--use_weighted_sampler`), and the Effective Number of Samples weighting scheme (Cui et al., 2019) calibrated for DFER datasets.
- **Staged Unfreezing:** Prevents catastrophic forgetting and early overfitting by freezing the transformer backbone for the first few epochs (`--freeze_epochs`), forcing the classification head to adapt first.
- **Robust Evaluation:** Built-in support for k-fold cross-validation (`--n_folds`) to provide defensible mean ± std metrics for minority classes, plus an option to evaluate only the single highest-scoring clip per video (`--test_top_clip_only`).
- **Train-Time Augmentation:** Configurable temporal jitter, color jitter, and horizontal flipping (`--augment`) to prevent memorization of oversampled minority classes.

### Training Examples

Both training scripts automatically download the necessary Hugging Face backbones if they are not cached locally.

**Standard Fine-Tuning (TimeSformer):**

```bash
python src/model_training/train_timesformer.py \
    --data_dir ./data/distilled_clips/max_emotion \
    --save_dir ./checkpoints/timesformer \
    --epochs 50 \
    --batch_size 16 \
    --freeze_epochs 3
```

**Training with Heavy Imbalance & Augmentation (ViViT):**

```bash
python src/model_training/train_vivit.py \
    --data_dir ./data/distilled_clips/max_emotion \
    --save_dir ./checkpoints/vivit \
    --class_weight_scheme effective_number \
    --loss_type focal \
    --use_weighted_sampler \
    --augment \
    --freeze_epochs 5 \
    --n_folds 5
```

For a full list of hyperparameters (including custom learning rates for the backbone vs. head, gradient accumulation, and learning rate scheduling), run `python src/model_training/train_timesformer.py --help`.

## 📦 Pretrained Models

Checkpoints for all trained models are provided below, organized by dataset, architecture, and distillation strategy.

| Dataset | Architecture | Distillation Strategy | Download |
|---|---|---|---|
| **DFEW** | **TimeSformer** | Max-Emotion (proposed) | [Google Drive](https://drive.google.com/file/d/1fUPRBqq1mNY6_6hOg8jUCiuBc_YAjea5/view?usp=drive_link) |
| | | Center Clips | [Google Drive](https://drive.google.com/file/d/1ShChZJ-wzc5gV3LNvBF2n0EBB0cfAW37/view?usp=drive_link) |
| | | Random Sampling | [Google Drive](https://drive.google.com/file/d/1PhTzc048ugmDffGirXJW4AwPPGULSPBV/view?usp=drive_link) |
| | **ViViT** | Max-Emotion (proposed) | [Google Drive](https://drive.google.com/file/d/123MuVwaHfYxzAyJ9LLRMzZkOaPkKYdgx/view?usp=drive_link) |
| | | Center Clips | [Google Drive](https://drive.google.com/file/d/1QNH8h7ekEs29zd11cq67Kj3iOAdFX6Xu/view?usp=drive_link) |
| | | Random Sampling | [Google Drive](https://drive.google.com/file/d/1Mnfy0HJpf0pI4f8wOor9lLToLQD19gr1/view?usp=drive_link) |
| **EMOGNITION** | **TimeSformer** | Max-Emotion (proposed) | [Google Drive](https://drive.google.com/file/d/1MwshzD45fq0ECWfhoIL3yh2KppzOlhFI/view?usp=drive_link) |
| | | Center Clips | [Google Drive](https://drive.google.com/file/d/12mg0FbBdMLH7fKwvEHzTBOFjb8A1a2Hm/view?usp=drive_link) |
| | | Random Sampling | [Google Drive](https://drive.google.com/file/d/1hP-s17Zhml_cA76i7O-cTBI-PA9AQj95/view?usp=drive_link) |
| | **ViViT** | Max-Emotion (proposed) | [Google Drive](https://drive.google.com/file/d/16_k4jWuKSxAFLThImSSyjJvBQJIXOtfL/view?usp=drive_link) |
| | | Center Clips | [Google Drive](https://drive.google.com/file/d/1ce9GcwKgcDfxBs4dnU4or_Y96IfkgVQf/view?usp=drive_link) |
| | | Random Sampling | [Google Drive](https://drive.google.com/file/d/1kgn6sSier15GSJ7t8Aywng-zgMtwMqhA/view?usp=drive_link) |

## 📝 Citation

If you use this codebase or the emotion-guided temporal distillation methodology, please cite:

```bibtex
@article{roodaki2026emotion,
  title={Emotion-Guided Data Distillation for Spatio-Temporal Feature Learning in Video Transformer-Based Dynamic Facial Expression Recognition},
  author={Roodaki, AmirHossein and Sotoodeh, Mahmood and Moosavi, Mohammad R. and Mbilinyi, Ashery},
  year={2026}
}
```
