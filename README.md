# Transformer-VFER: Emotion-Guided Data Distillation

Official PyTorch implementation for "Emotion-Guided Data Distillation for Spatio-Temporal Feature Learning in Video Transformer-Based Facial Expression Recognition".

This repository provides a modular, end-to-end pipeline for processing raw video datasets, distilling them into high-signal expressive segments using DeepFace, and fine-tuning Video Vision Transformers (ViViT, TimeSformer) for nuanced facial expression recognition.

## 🚀 Installation & Setup

We recommend using an Anaconda environment to manage dependencies.

```bash
# Clone the repository
git clone https://github.com/Roodaki/Transformer-VFER.git
cd Transformer-VFER

# Create and activate environment
conda create -n vfer python=3.9
conda activate vfer

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install pipeline dependencies
pip install deepface mediapipe opencv-python pandas numpy
```

## 🗂️ Codebase Architecture

The repository is structured to separate the data distillation pipeline from the deep learning model architectures.

```
├── models/                     # PyTorch Transformer Architectures
│   ├── vivit/                  # ViViT: Factorised Encoder implementation
│   ├── timesformer/            # TimeSformer: Divided Space-Time Attention
│   └── videomae/               # VideoMAE backbone (alternative backbone)
├── utils/                      # Data Distillation Pipeline Scripts
│   ├── extract_faces_mediapipe.py # MTCNN/MediaPipe face detection and ROI cropping
│   ├── analyze_videos.py       # Generates frame-by-frame emotion probability vectors (DeepFace)
│   ├── trim_videos.py          # Salience scoring (sliding window) and top-k clip extraction
│   ├── rotate_videos.bat       # Geometric normalization (90° clockwise)
│   ├── get_videos_length.py    # Dataset video duration analysis
│   └── organize_videos.py      # Dataset structuring
└── README.md
```

## ⚙️ Data Distillation Pipeline

Naturalistic datasets (like Emognition) contain long, neutral video sequences that dilute learning. Run the following pipeline to distill raw videos into dense, emotion-rich tensors.

### 1. Geometric Normalization & Face ROI Extraction

Correct orientation and isolate the facial region to eliminate background noise.

```bash
# Windows users: Run ./utils/rotate_videos.bat first if needed
python utils/extract_faces_mediapipe.py \
    --input_dir ./data/raw_videos \
    --output_dir ./data/cropped_faces
```

### 2. Emotion-Guided Frame Analysis

Pass the cropped frames through the teacher model (DeepFace) to generate temporal emotion probability vectors $P_t = \langle p_{t,1}, p_{t,2}, \ldots, p_{t,c} \rangle$.

```bash
python utils/analyze_videos.py \
    --data_dir ./data/cropped_faces \
    --output ./data/csv_logs/
```

### 3. Temporal Distillation (Trimming)

Calculate moving-window salience scores and extract the top-$k$ most expressive clips. The output length must match the target transformer's expected frame count $L$.

```bash
# For ViViT (L=16 frames)
python utils/trim_videos.py --logs ./data/csv_logs/ --output_dir ./data/distilled_clips/ --clip_length 16

# For TimeSformer (L=8 frames)
python utils/trim_videos.py --logs ./data/csv_logs/ --output_dir ./data/distilled_clips/ --clip_length 8
```

## 🧠 Model Architectures & Training

The models/ directory contains standard PyTorch implementations of the video transformers. Input tensors must be shaped as [Batch, Channels, Frames, Height, Width].

### Model Technical Specs

| Architecture | Attention Scheme   | Input Tensor Shape   | Patch Size | Params | Pre-training                |
| ------------ | ------------------ | -------------------- | ---------- | ------ | --------------------------- |
| TimeSformer  | Divided Space-Time | [B, 3, 8, 224, 224]  | 16x16      | 122M   | ImageNet-21K                |
| ViViT        | Factorised Encoder | [B, 3, 16, 224, 224] | 16x16      | 115M   | ImageNet-21K + Kinetics-400 |

### Fine-Tuning Configuration

Models are fine-tuned by freezing the pretrained backbone and training only the final linear classification head.

**Hyperparameters:**

- Optimizer: AdamW
- Learning Rate: 5e-5
- Weight Decay: 0.01
- Scheduler: ReduceLROnPlateau (Patience: 3, Factor: 0.1, Min LR: 1e-7)
- Loss Function: Cross-Entropy Loss
- Batch Size: 8
- Early Stopping: Patience of 15 epochs (Min Delta: 0.001) based on Validation F1-Score.

## 📊 Evaluation Results

Evaluated on the 10-class Emognition dataset (3-fold subject-exclusive cross-validation). Distillation yields a ~15% performance gain over random frame sampling.

| Architecture | Overall Accuracy | Macro F1-Score |
| ------------ | ---------------- | -------------- |
| TimeSformer  | 98.61%           | 98.52%         |
| ViViT        | 99.59%           | 99.64%         |

## 📝 Citation

If you utilize this codebase or our data distillation methodology, please cite:

```bibtex
@article{roodaki2024emotion,
  title={Emotion-Guided Data Distillation for Spatio-Temporal Feature Learning in Video Transformer-Based Facial Expression Recognition},
  author={Roodaki, AmirHossein and Sotoodeh, Mahmood and Moosavi, Mohammad R. and Mbilinyi, Ashery},
  year={2024}
}
```
