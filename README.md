# Emotion-Guided Data Distillation for Spatio-Temporal Feature Learning in Video Transformer-Based Facial Expression Recognition

[![DOI](https://zenodo.org/badge/1073745175.svg)](https://doi.org/10.5281/zenodo.21793814)

This repository provides a modular, end-to-end pipeline for processing raw video datasets, distilling them into high-signal expressive temporal segments using emotion-based salience scoring, and fine-tuning video transformer models for dynamic facial expression recognition (DFER).

<img width="637" height="961" alt="image" src="https://github.com/user-attachments/assets/122c092b-f720-48e3-98e5-12d6318925a9" />

## Requirements

We recommend using an Anaconda environment to manage dependencies.

```setup
# Clone the repository
git clone https://github.com/Roodaki/Temporal-Distillation-DFER.git
cd Temporal-Distillation-DFER

# Create and activate environment
conda create -n dfer python=3.9 -y
conda activate dfer

# Install PyTorch (adjust the CUDA version according to your system if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install pipeline dependencies
pip install -r requirements.txt
```

Pretrained Hugging Face backbones and training checkpoints are written locally at run time and are not tracked in Git.

## Training

Naturalistic facial expression datasets often contain long videos with neutral, low-signal, or redundant frames. Before fine-tuning a video transformer, raw videos are run through a distillation pipeline that extracts face ROIs, computes frame-by-frame emotion salience, and distills each video into compact, high-signal clips.

### 1. Face ROI Extraction

Use MediaPipe to crop facial regions and reduce background noise.

```train
python src/video_preprocessing/extract_faces_mediapipe.py \
    --input_dir ./data/raw_videos \
    --output_dir ./data/cropped_faces
```

### 2. Emotion-Guided Frame/Video Analysis

Generate emotion probability logs (`P_t = <p_t,1, ..., p_t,c>`) for each frame using DeepFace.

```train
python src/video_preprocessing/analyze_videos.py \
    --data_dir ./data/cropped_faces \
    --output ./data/csv_logs
```

### 3. Temporal Distillation & Baselines

A single consolidated script handles emotion-guided distillation and baseline extraction (center and random clips). To handle inherently imbalanced DFER datasets, this script also features built-in class balancing strategies (`--balance_mode` and `--k_strategy`) to prevent over-represented classes from dominating the distilled dataset.

**Emotion-guided distillation (max-emotion):** extracts high-salience temporal clips based on the emotion probability logs. Candidates are dynamically thresholded against the video's own best window to ensure weak-but-genuine expressions are preserved.

```train
python src/video_preprocessing/trim_videos.py \
    --mode max_emotion \
    --source_dir ./data/cropped_faces \
    --logs ./data/csv_logs \
    --output_dir ./data/distilled_clips \
    --clip_length 16 \
    --balance_mode global_topk
```

**Generating baselines (center & random):** these do not require the emotion CSV logs.

```train
python src/video_preprocessing/trim_videos.py \
    --mode random \
    --source_dir ./data/cropped_faces \
    --output_dir ./data/distilled_clips \
    --clip_length 16 \
    --k_strategy auto_balance
```

Passing `--mode both` extracts the emotion-guided dataset and both baselines simultaneously into three separate subdirectories (`max_emotion`, `center_clips`, and `random`).

### 4. Utilities & Visualization

```train
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

### 5. Model Training

This repository supports transformer-based video models for facial expression recognition, specifically TimeSformer and ViViT. Both scripts automatically download the necessary Hugging Face backbones if they are not cached locally.

The training pipeline is engineered to handle the unique challenges of naturalistic DFER datasets — specifically severe class imbalance and train/test data leakage:

- **Strict Anti-Leakage Splitting:** automatically extracts the original video ID from distilled clips and uses `StratifiedGroupKFold` to guarantee that clips from the same source video never cross train/val/test boundaries.
- **Advanced Imbalance Handling:** supports raw inverse weighting, Focal Loss (`--loss_type focal`), batch-level oversampling (`--use_weighted_sampler`), and the Effective Number of Samples weighting scheme (Cui et al., 2019) calibrated for DFER datasets.
- **Staged Unfreezing:** prevents catastrophic forgetting and early overfitting by freezing the transformer backbone for the first few epochs (`--freeze_epochs`), forcing the classification head to adapt first.
- **Train-Time Augmentation:** configurable temporal jitter, color jitter, and horizontal flipping (`--augment`) to prevent memorization of oversampled minority classes.

```train
# Standard fine-tuning (TimeSformer)
python src/model_training/train_timesformer.py \
    --data_dir ./data/distilled_clips/max_emotion \
    --save_dir ./checkpoints/timesformer \
    --epochs 50 \
    --batch_size 16 \
    --freeze_epochs 3

# Training with heavy imbalance & augmentation (ViViT)
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

## Evaluation

There is no separate evaluation script — evaluation is built into the training scripts. Each script carves out a group-aware held-out test split (`--test_split`, default `0.2`, split via `StratifiedGroupKFold` so clips from the same source video never leak across the split), then evaluates the trained model on it:

```eval
python src/model_training/train_timesformer.py \
    --data_dir ./data/distilled_clips/max_emotion \
    --save_dir ./checkpoints/timesformer \
    --test_split 0.2 \
    --n_folds 5 \
    --test_top_clip_only
```

- `--n_folds` (default `1`) runs `k`-fold cross-validation over the training pool and reports mean ± std metrics across folds, in addition to the single held-out test evaluation.
- `--test_top_clip_only` restricts the held-out test set to only the highest-salience clip per source video, for a stricter per-video evaluation.

Results are written to `--save_dir`:
- `final_class_performance.csv` — per-class precision/recall/F1 on the held-out test set.
- `cv_class_performance_summary.csv` — cross-fold mean ± std class performance (only when `--n_folds > 1`).
- `final_timesformer_summary.txt` / `final_vivit_summary.txt` — overall run summary.

## Results

>📋 Include a table of results from your paper, and link back to the leaderboard for clarity and context. Quantitative benchmark numbers (accuracy / F1 per dataset) are not yet published here — see the paper for reported results.

### In-The-Lab "Emognition" Dataset

| Model Name | Top-1 Accuracy | F1-Score | UAR | WAR |
|------------|----------------|----------|-----|-----|
| ViViT | 82.92% | 83.11% | 81.59% | 82.92% |
| TimeSformer | 81.54% | 80.86% | 78.99% | 81.54% |

### In-The-Wild "DFEW" Dataset

| Model Name | Top-1 Accuracy | F1-Score | UAR | WAR |
|------------|----------------|----------|-----|-----|
| ViViT | 90.93% | 81.31% | 77.32% | 90.93% |
| TimeSformer | 89.51% | 77.69% | 74.66% | 89.51% |

### Pretrained Models

Checkpoints for all trained models are provided below, organized by dataset, architecture, and distillation strategy.

| Dataset        | Architecture    | Distillation Strategy  | Download                                                                                                    |
| -------------- | --------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------ |
| **DFEW**       | **TimeSformer** | Max-Emotion (proposed) | [Google Drive](https://drive.google.com/file/d/1fUPRBqq1mNY6_6hOg8jUCiuBc_YAjea5/view?usp=drive_link)        |
|                |                 | Center Clips           | [Google Drive](https://drive.google.com/file/d/1ShChZJ-wzc5gV3LNvBF2n0EBB0cfAW37/view?usp=drive_link)        |
|                |                 | Random Sampling        | [Google Drive](https://drive.google.com/file/d/1PhTzc048ugmDffGirXJW4AwPPGULSPBV/view?usp=drive_link)        |
|                | **ViViT**       | Max-Emotion (proposed) | [Google Drive](https://drive.google.com/file/d/123MuVwaHfYxzAyJ9LLRMzZkOaPkKYdgx/view?usp=drive_link)        |
|                |                 | Center Clips           | [Google Drive](https://drive.google.com/file/d/1QNH8h7ekEs29zd11cq67Kj3iOAdFX6Xu/view?usp=drive_link)        |
|                |                 | Random Sampling        | [Google Drive](https://drive.google.com/file/d/1Mnfy0HJpf0pI4f8wOor9lLToLQD19gr1/view?usp=drive_link)        |
| **EMOGNITION** | **TimeSformer** | Max-Emotion (proposed) | [Google Drive](https://drive.google.com/file/d/1MwshzD45fq0ECWfhoIL3yh2KppzOlhFI/view?usp=drive_link)        |
|                |                 | Center Clips           | [Google Drive](https://drive.google.com/file/d/12mg0FbBdMLH7fKwvEHzTBOFjb8A1a2Hm/view?usp=drive_link)        |
|                |                 | Random Sampling        | [Google Drive](https://drive.google.com/file/d/1hP-s17Zhml_cA76i7O-cTBI-PA9AQj95/view?usp=drive_link)        |
|                | **ViViT**       | Max-Emotion (proposed) | [Google Drive](https://drive.google.com/file/d/16_k4jWuKSxAFLThImSSyjJvBQJIXOtfL/view?usp=drive_link)        |
|                |                 | Center Clips           | [Google Drive](https://drive.google.com/file/d/1ce9GcwKgcDfxBs4dnU4or_Y96IfkgVQf/view?usp=drive_link)        |
|                |                 | Random Sampling        | [Google Drive](https://drive.google.com/file/d/1kgn6sSier15GSJ7t8Aywng-zgMtwMqhA/view?usp=drive_link)        |

## Codebase Architecture

The repository is divided into the core pipeline (`src/`) and analysis tools (`utils/`). Pretrained Hugging Face backbones and training checkpoints are written locally at run time and are not tracked in Git.

```text
.
├── src/
│   ├── model_training/            # Video transformer fine-tuning scripts
│   │   ├── train_timesformer.py
│   │   ├── train_timesformer_leakage.py
│   │   ├── train_vivit.py
│   │   └── train_vivit_leakage.py
│   └── video_preprocessing/       # Pipeline for data extraction & distillation
│       ├── analyze_videos.py
│       ├── extract_faces_mediapipe.py
│       ├── organize_videos.py
│       ├── rotate_videos.bat
│       └── trim_videos.py         # Consolidated emotion-guided and random trimming
├── utils/
│   ├── generate_figures/          # Visualization utilities
│   │   └── draw_emotion_segmentation_figure.py
│   └── video_processing/          # Dataset statistics and analysis tools
│       ├── analyze_class_score_distributions.py
│       ├── count_clips.py
│       └── get_videos_length.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you use this codebase or the emotion-guided temporal distillation methodology, please cite:

```bibtex
@article{roodaki2026emotion,
  title={Emotion-Guided Data Distillation for Spatio-Temporal Feature Learning in Video Transformer-Based Dynamic Facial Expression Recognition},
  author={Roodaki, AmirHossein and Sotoodeh, Mahmood and Moosavi, Mohammad R. and Mbilinyi, Ashery},
  year={2026}
}
```

## Contributing

This project is licensed under the [MIT License](./LICENSE). Contributions are welcome — please open an issue to discuss proposed changes, or submit a pull request.
