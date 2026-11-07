# Emotion-Guided Data Distillation for Spatio-Temporal Feature Learning in Video Transformer-Based Dynamic Facial Expression Recognition

[![DOI](https://zenodo.org/badge/1073745175.svg)](https://doi.org/10.5281/zenodo.21793814)

This repository provides a modular, end-to-end pipeline for processing raw video datasets, distilling them into high-signal expressive temporal segments using emotion-based salience scoring, and fine-tuning video transformer models for dynamic facial expression recognition (DFER).

<img width="720" height="1133" alt="shematic" src="https://github.com/user-attachments/assets/679b0bf5-c19a-4611-b48a-66134f4f4091" />


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

Naturalistic facial expression datasets often contain long videos with neutral, low-signal, or redundant frames. Before fine-tuning a video transformer, raw videos are run through a distillation pipeline that normalizes orientation, isolates the face, computes frame-by-frame emotion probabilities with an off-the-shelf teacher model (DeepFace), and distills each video into compact, high-signal clips. The teacher is used only for temporal localization: clips keep the dataset's original video-level labels, and teacher predictions are never used as training labels.

### 1. Geometric Normalization & Face ROI Extraction

For Emognition, first rotate every video 90° clockwise to correct the recording orientation (Windows batch script, requires FFmpeg; run it inside the folder that contains the videos):

```train
src\video_preprocessing\rotate_videos.bat
```

Then use MTCNN to detect and crop the facial region in every frame, removing background noise:

```train
python src/video_preprocessing/extract_faces_mtcnn.py \
    --input_dir ./data/raw_videos \
    --output_dir ./data/cropped_faces
```

### 2. Frame-by-Frame Emotion Analysis

Generate emotion probability logs (`P_t = <p_t,1, ..., p_t,c>`) for each frame using DeepFace.

```train
python src/video_preprocessing/analyze_videos.py \
    --data_dir ./data/cropped_faces \
    --output ./data/csv_logs
```

### 3. Temporal Distillation & Baselines

A single consolidated script handles emotion-guided distillation and the baseline clip extractors (center and random clips). The clip length `L` must match the model input: **16 frames for ViViT, 8 frames for TimeSformer**.

**Emotion-guided distillation (max-emotion):** a window of `L` frames slides over each video (stride 1) and is scored by the mean teacher probability over its frames. Windows with no expressive content are discarded, and the remaining windows are selected greedily in descending score order while removing any window that overlaps an already-selected one (temporal non-maximum suppression). For every video the script writes two kinds of clips:

- **Training clips** (`<video>_max_emotion_<k>.mp4`, numbered by salience): scored with the probability of the video's label mapped to the teacher's classes (e.g. `amusement`, `liking`, `enthusiasm` → `happy`; `awe` → `surprise`).
- **One label-agnostic test clip** (`<video>_test_1.mp4`): scored with the mean of `1 - p_neutral`, without access to the label. If a video has no expressive window at all, the window with the highest sustained `p_neutral` is used. This is the same procedure used for any unseen video at deployment time.

```train
# ViViT (L = 16); use --clip_length 8 for TimeSformer
python src/video_preprocessing/trim_videos.py \
    --mode max_emotion \
    --source_dir ./data/cropped_faces \
    --logs ./data/csv_logs \
    --output_dir ./data/distilled_clips_16 \
    --clip_length 16
```

The number of training clips a video yields depends on its expressive content (long, largely neutral videos yield only a few). To handle class imbalance (e.g. DFEW), pass `--k_strategy auto_balance --base_k <K>`: videos from smaller classes then contribute proportionally more training clips (`K_class = round(K · max_class_count / class_count)`, similar to weighted oversampling). On balanced datasets such as Emognition this has no effect.

**Generating baselines (center & random):** these do not require the emotion CSV logs. Center sampling takes the `L` consecutive frames centered on the video's temporal midpoint; random sampling takes a random window of `L` consecutive frames.

```train
python src/video_preprocessing/trim_videos.py \
    --mode random \
    --source_dir ./data/cropped_faces \
    --output_dir ./data/distilled_clips_16 \
    --clip_length 16
```

Passing `--mode both` extracts the emotion-guided dataset and both baselines simultaneously into three separate subdirectories (`max_emotion`, `center_clips`, and `random`).

### 4. Utilities & Visualization

```train
# Visualize emotion-guided temporal segmentation
python utils/generate_figures/draw_emotion_segmentation_figure.py --root_dir ./data/csv_logs

# Organize flat datasets into class subfolders
python src/video_preprocessing/organize_videos.py \
    --source_dir ./data/distilled_clips_16/max_emotion \
    --output_dir ./data/distilled_organized

# Build a fold file from the official DFEW five-fold split
python utils/video_processing/make_fold_file.py \
    --test_csvs ./DFEW/test_single_labeled/set_1.csv ./DFEW/test_single_labeled/set_2.csv \
                ./DFEW/test_single_labeled/set_3.csv ./DFEW/test_single_labeled/set_4.csv \
                ./DFEW/test_single_labeled/set_5.csv \
    --output ./data/dfew_folds.csv

# Dataset analysis tools
python utils/video_processing/get_videos_length.py --dataset_root ./data/cropped_faces
python utils/video_processing/count_clips.py --dataset_root ./data/distilled_clips_16
python utils/video_processing/analyze_class_score_distributions.py --logs ./data/csv_logs
```

### 5. Model Training

Two pre-trained video transformers are fine-tuned, using the base variant of each, with a linear classification head added on top:

| Model | Clip length (L) | Default checkpoint |
|-------|-----------------|--------------------|
| ViViT-Base | 16 | `google/vivit-b-16x2-kinetics400` |
| TimeSformer-Base | 8 | `facebook/timesformer-base-finetuned-k400` |

Both scripts automatically download the Hugging Face backbones if they are not cached locally. Fine-tuning uses a freeze-unfreeze strategy: the backbone is frozen for the first `--freeze_epochs` epochs so only the classification head trains, then the whole model is unfrozen.

The script defaults match the hyperparameter configuration used for all models:

| Hyperparameter | Value |
|----------------|-------|
| Batch size | 8 (Emognition), 4 (DFEW) |
| Epochs (maximum) | 300 |
| Optimizer | AdamW, learning rate 5e-5, weight decay 0.01 |
| Loss | Focal Loss |
| LR scheduler | ReduceLROnPlateau on validation loss (patience 3, factor 0.1, min LR 1e-7) |
| Early stopping | Validation F1, patience 15, min delta 0.001 |
| Fine-tuning | Freeze-unfreeze |

```train
# Emognition: subject-independent 3-fold cross-validation (ViViT)
python src/model_training/train_vivit.py \
    --data_dir ./data/distilled_clips_16/max_emotion \
    --save_dir ./checkpoints/vivit_emognition \
    --num_classes 10 \
    --batch_size 8 \
    --n_folds 3 \
    --group_by subject

# DFEW: official 5-fold protocol (TimeSformer)
python src/model_training/train_timesformer.py \
    --data_dir ./data/distilled_clips_8/max_emotion \
    --save_dir ./checkpoints/timesformer_dfew \
    --num_classes 7 \
    --batch_size 4 \
    --fold_file ./data/dfew_folds.csv \
    --group_by video \
    --val_split 0.15
```

Additional options cover class-imbalance handling (`--class_weight_scheme`, including the Effective Number of Samples weighting of Cui et al., 2019; `--use_weighted_sampler`), train-time augmentation (`--augment`), separate backbone/head learning rates, and gradient accumulation. Run `python src/model_training/train_timesformer.py --help` for the full list.

## Evaluation

Evaluation is built into the training scripts and follows a video-level cross-validation protocol:

- **Folds.** Every source video belongs to exactly one test fold. With `--fold_file`, folds come from a predefined assignment (e.g. the official DFEW split). Otherwise `--n_folds` stratified folds are built with `StratifiedGroupKFold`. `--group_by subject` makes the folds subject-independent: the subject id is parsed from the video name with `--subject_regex`, which defaults to the token before the first underscore (Emognition). `--group_by video` keeps only the clips of one video together.
- **Validation.** In each fold, a group-disjoint fraction of the training pool (`--val_split`) is held out for early stopping and learning-rate scheduling.
- **No leakage.** All clips of a video, and with `--group_by subject` all videos of a subject, stay in a single partition. This is asserted at run time.
- **One clip per test video.** Each test video is represented by its label-agnostic `<video>_test_1` clip. For baseline datasets without test clips, its `_1` clip is used instead.
- **Pooling.** Test predictions from all folds are pooled to compute the reported metrics.

Results are written to `--save_dir`:
- `final_class_performance.csv` — pooled per-class precision/recall/F1 plus accuracy, WAR, UAR, macro and weighted F1.
- `test_confusion_matrix.png` — pooled normalized confusion matrix.
- `final_class_performance_fold<k>.csv`, `test_confusion_matrix_fold<k>.png`, `test_predictions_fold<k>.csv` — per-fold results and predictions.
- `cv_class_performance_summary.csv` — per-class mean ± std across folds.
- `final_timesformer_summary.txt` / `final_vivit_summary.txt` — overall run summary.

To resume an interrupted run, pass `--resume_from <save_dir>/last_checkpoint_fold<k>.pth`. Earlier folds are then restored from their saved predictions.

## Results

### In-The-Lab "Emognition" Dataset

Subject-independent three-fold cross-validation, pooled over all folds.

| Model Name | Accuracy | Macro F1-Score |
|------------|----------|----------------|
| ViViT | 82.82% | 82.82% |
| TimeSformer | 81.54% | 81.40% |

Ablation of the temporal sampling strategy (same backbones and fine-tuning):

| Model | Sampling Strategy | Accuracy | Macro F1-Score |
|-------|-------------------|----------|----------------|
| ViViT | Random | 13.59% | 10.91% |
| | Center-clip | 33.33% | 32.61% |
| | Emotion-guided distillation | **82.82%** | **82.82%** |
| TimeSformer | Random | 12.05% | 10.76% |
| | Center-clip | 37.44% | 35.84% |
| | Emotion-guided distillation | **81.54%** | **81.40%** |

### In-The-Wild "DFEW" Dataset

Official seven-class, five-fold protocol, pooled over all folds.

| Model Name | WAR | UAR | Macro F1-Score |
|------------|-----|-----|----------------|
| ViViT | 90.93% | 77.32% | 81.31% |
| TimeSformer | 89.51% | 74.66% | 77.69% |

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
│   │   ├── evaluation_protocol.py # Cross-validation folds, partitioning & metrics
│   │   ├── train_timesformer.py
│   │   ├── train_timesformer_leakage.py
│   │   ├── train_vivit.py
│   │   └── train_vivit_leakage.py
│   └── video_preprocessing/       # Pipeline for data extraction & distillation
│       ├── analyze_videos.py
│       ├── extract_faces_mtcnn.py
│       ├── organize_videos.py
│       ├── rotate_videos.bat      # 90° clockwise geometric normalization
│       └── trim_videos.py         # Emotion-guided distillation, center & random clips
├── utils/
│   ├── generate_figures/          # Visualization utilities
│   │   └── draw_emotion_segmentation_figure.py
│   └── video_processing/          # Dataset statistics and analysis tools
│       ├── analyze_class_score_distributions.py
│       ├── count_clips.py
│       ├── get_videos_length.py
│       └── make_fold_file.py      # Fold file from official split lists (e.g. DFEW)
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
