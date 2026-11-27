# Emotion-Guided Data Distillation for Spatio-Temporal Feature Learning in Video Transformer-Based Dynamic Facial Expression Recognition

[![DOI](https://zenodo.org/badge/1073745175.svg)](https://doi.org/10.5281/zenodo.21793814)

Code for the paper: video preprocessing, emotion-guided temporal distillation, and fine-tuning of video transformers for dynamic facial expression recognition (DFER).

<img width="720" height="1133" alt="schematic" src="https://github.com/user-attachments/assets/679b0bf5-c19a-4611-b48a-66134f4f4091" />


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

Before fine-tuning, raw videos are processed in four steps: geometric normalization, face cropping, frame-by-frame emotion analysis with DeepFace, and extraction of fixed-length clips. The method is described in the paper.

### 1. Geometric Normalization & Face ROI Extraction

For Emognition, first rotate every video 90° clockwise to correct the recording orientation (Windows batch script, requires FFmpeg; run it inside the folder that contains the videos):

```train
src\video_preprocessing\rotate_videos.bat
```

Then detect and crop the face in every frame with MTCNN:

```train
python src/video_preprocessing/extract_faces_mtcnn.py \
    --input_dir ./data/raw_videos \
    --output_dir ./data/cropped_faces
```

### 2. Frame-by-Frame Emotion Analysis

Compute per-frame emotion probabilities with DeepFace:

```train
python src/video_preprocessing/analyze_videos.py \
    --data_dir ./data/cropped_faces \
    --output ./data/csv_logs
```

### 3. Temporal Distillation & Baselines

`trim_videos.py` extracts the clips for the proposed emotion-guided distillation and for the two baselines (center-clip and random sampling). The clip length `L` is **16 frames for ViViT and 8 frames for TimeSformer**.

**Emotion-guided distillation** (`--mode max_emotion`) writes training clips (`<video>_max_emotion_<k>.mp4`) and one test clip per video (`<video>_test_1.mp4`), selected without the video's label. Class balancing of the training clips (`--k_strategy auto_balance`) was used for both datasets.

```train
# ViViT (L = 16); use --clip_length 8 for TimeSformer
python src/video_preprocessing/trim_videos.py \
    --mode max_emotion \
    --source_dir ./data/cropped_faces \
    --logs ./data/csv_logs \
    --output_dir ./data/distilled_clips_16 \
    --clip_length 16
```

**Baselines** (`--mode center_clips` or `--mode random`) do not need the emotion logs:

```train
python src/video_preprocessing/trim_videos.py \
    --mode random \
    --source_dir ./data/cropped_faces \
    --output_dir ./data/distilled_clips_16 \
    --clip_length 16
```

`--mode both` writes all three into separate subdirectories (`max_emotion`, `center_clips`, `random`).

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
python utils/video_processing/count_clips.py --output_dir ./data/distilled_clips_16
python utils/video_processing/analyze_class_score_distributions.py --logs ./data/csv_logs
```

### 5. Model Training

Two pre-trained video transformers (base variants) are fine-tuned with a linear classification head:

| Model | Clip length (L) | Default checkpoint |
|-------|-----------------|--------------------|
| ViViT-Base | 16 | `google/vivit-b-16x2-kinetics400` |
| TimeSformer-Base | 8 | `facebook/timesformer-base-finetuned-k400` |

The backbones are downloaded from Hugging Face on first use. The script defaults follow the hyperparameters in the paper:

| Hyperparameter | Value |
|----------------|-------|
| Batch size | 8 (Emognition), 4 (DFEW) |
| Epochs (maximum) | 300 |
| Optimizer | AdamW, learning rate 5e-5, weight decay 0.01 |
| Loss | Focal Loss |
| LR scheduler | ReduceLROnPlateau on validation loss (patience 3, factor 0.1, min LR 1e-7) |
| Early stopping | Validation F1, patience 15, min delta 0.001 |
| Fine-tuning | Freeze-unfreeze (`--freeze_epochs`) |

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

Run a script with `--help` to list all options.

## Evaluation

Evaluation runs at the end of training and follows the protocol described in the paper:

- **Folds.** Emognition uses subject-independent folds (`--n_folds 3 --group_by subject`). DFEW uses the official five-fold split (`--fold_file`).
- **Validation.** In each fold, a held-out part of the training data (`--val_split`) is used for early stopping and learning-rate scheduling.
- **Partitions.** All clips of a video, and on Emognition all videos of a subject, stay in a single partition.
- **Test clips.** Each test video is represented by one clip.
- **Metrics.** Test predictions from all folds are pooled. The sample standard deviation across the test folds and 95% class-stratified bootstrap confidence intervals (10,000 resamples) are reported alongside.

Results are written to `--save_dir`:
- `final_class_performance.csv`: pooled per-class and overall metrics.
- `test_confusion_matrix.png`: pooled normalized confusion matrix.
- `final_class_performance_fold<k>.csv`, `test_confusion_matrix_fold<k>.png`, `test_predictions_fold<k>.csv`: per-fold results.
- `cv_class_performance_summary.csv`: mean ± standard deviation across folds.
- `pooled_metrics_ci.csv`: confidence intervals of the pooled metrics.
- `final_timesformer_summary.txt` / `final_vivit_summary.txt`: run summary.

To resume an interrupted run, pass `--resume_from <save_dir>/last_checkpoint_fold<k>.pth`.

## Results

### In-The-Lab "Emognition" Dataset

Subject-independent three-fold cross-validation. Each cell gives the pooled value ± the standard deviation across folds, with the 95% confidence interval in brackets.

| Model Name | Accuracy | Macro F1-Score |
|------------|----------|----------------|
| ViViT | 82.82% ± 3.1 [78.97, 86.41] | 82.82% ± 3.0 [78.98, 86.42] |
| TimeSformer | 81.54% ± 3.1 [77.69, 85.13] | 81.40% ± 3.1 [77.46, 85.12] |

Ablation of the temporal sampling strategy:

| Model | Sampling Strategy | Accuracy | Macro F1-Score |
|-------|-------------------|----------|----------------|
| ViViT | Random | 13.59% | 10.91% |
| | Center-clip | 33.33% | 32.61% |
| | Emotion-guided distillation | **82.82%** | **82.82%** |
| TimeSformer | Random | 12.05% | 10.76% |
| | Center-clip | 37.44% | 35.84% |
| | Emotion-guided distillation | **81.54%** | **81.40%** |

### In-The-Wild "DFEW" Dataset

Official seven-class, five-fold protocol. Same notation as above.

| Model Name | WAR | UAR | Macro F1-Score |
|------------|-----|-----|----------------|
| ViViT | 87.30% ± 0.9 [86.75, 87.86] | 77.75% ± 1.4 [76.56, 78.98] | 80.57% ± 1.6 [79.13, 81.94] |
| TimeSformer | 86.20% ± 0.9 [85.62, 86.77] | 74.95% ± 1.3 [73.96, 75.96] | 76.95% ± 1.7 [75.54, 78.29] |

### Pretrained Models

Trained checkpoints, by dataset, architecture and sampling strategy:

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

The repository is divided into the core pipeline (`src/`) and analysis tools (`utils/`).

```text
.
├── src/
│   ├── model_training/            # Video transformer fine-tuning scripts
│   │   ├── evaluation_protocol.py # Cross-validation folds, partitioning & metrics
│   │   ├── train_timesformer.py
│   │   └── train_vivit.py
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
