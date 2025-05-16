# EEG Dementia Classification using Vision Transformers and DINO

This project implements a deep learning pipeline for classifying EEG signals using a 3D Vision Transformer (ViT3D) architecture. The model leverages DINO-pretrained ViT weights (originally trained on 2D images), inflated to handle 3D EEG inputs, and supports uncertainty quantification through split-conformal prediction.

## 🧠 Project Overview

Electroencephalogram (EEG) data is inherently spatiotemporal. This project repurposes state-of-the-art vision transformer backbones to classify dementia-relevant EEG patterns. The architecture integrates:

- A 3D Vision Transformer (ViT3D) with patch embedding inflation.
- DINO (self-supervised) pretrained ViT weights.
- Conformal prediction for confidence-calibrated decision-making.

## 📁 Directory Structure

src/
├── dataset_with_aug.py # EEG dataset loader with data augmentation
├── dino3d.py # Vision Transformer backbone adapted for 3D inputs
├── model.py # LightningModule wrapper for training and evaluation
├── preprocessing.py # EEG preprocessing utilities
├── simply_dataset.py # Minimal EEG dataset loader (no augmentation)
├── train.py # Training entry script using PyTorch Lightning
├── utils.py # Helper functions (e.g., metrics, logging)
LICENSE
README.md


## 🧩 Features

- ✅ Inflated 2D ViT weights (from DINO) to 3D for spatiotemporal EEG data
- ✅ Modular Lightning-based training pipeline
- ✅ EEG-specific data augmentations (SmoothTimeMask, BandstopFilter)
- ✅ CLS token-based classification with optional age embedding
- ✅ Split-conformal prediction for uncertainty estimation

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/eeg_vit_dino.git
cd eeg_vit_dino

# Create environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

Note: This project uses PyTorch Lightning, NumPy, SciPy, and Matplotlib. GPU acceleration (CUDA) is recommended for training.
📊 Dataset
EEG .npy files are expected to follow the structure:

Shape: (Channels, Electrodes, Timesteps) = (160, 19, 500)

Directory structure:

bash
Copy
Edit
/data/
  ├── train/
  ├── val/
  └── test/
Each subdirectory should contain EEG arrays and a corresponding participants.tsv file for metadata (e.g., age, labels).

🚀 Training
Modify training hyperparameters in train.py as needed.

bash
Copy
Edit
python src/train.py --data_dir /path/to/data --epochs 100 --batch_size 4
Key Arguments:
--data_dir: Path to EEG dataset

--pretrained: Use DINO weights (True/False)

--freeze_backbone: Whether to freeze ViT backbone

--lr: Learning rate

--use_age: Include age embeddings in the classifier

📈 Evaluation
Validation accuracy, loss, and calibration metrics are logged during training. Final classification metrics and conformal prediction coverage are available in the logs.

Use plot_logs() from utils.py to visualize training dynamics.

📉 Uncertainty Estimation
This project integrates split-conformal prediction to compute per-sample confidence sets. Calibration ensures nominal coverage (e.g., 90%) while maintaining discriminative power.

Uncertainty-related outputs include:

Prediction sets

Empirical coverage

False negative control via margin-based thresholds

🧪 Augmentation Techniques
Implemented in dataset_with_aug.py, including:

SmoothTimeMask: Smooth temporal dropout

BandstopFilter: Simulates sensor noise by filtering frequency bands

Augmentations are applied probabilistically during training.

🤝 Contributions
This research is part of an academic thesis exploring uncertainty quantification in EEG-based dementia detection using vision transformers.

Feel free to fork, raise issues, or contribute enhancements.

📄 License
This project is licensed under the terms of the MIT License.