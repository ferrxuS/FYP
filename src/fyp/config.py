import torch
from pathlib import Path

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Project Roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "dataset"
TRAIN_DIR = DATASET_ROOT / "train"
VALID_DIR = DATASET_ROOT / "valid"
SAVE_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset Paths (for extraction)
ZIP_PATH = PROJECT_ROOT / "data" / "TAO Checkpoint.v2-v0-bin.png-mask-semantic.zip"

# Model Configuration
BACKBONE = "vitl16"         # DINOv3 variant: vits16, vitb16, vitl16, vith16
NUM_CLASSES = 2             # Binary: background + defect
IMAGE_SIZE = 768            # must be divisible by PATCH_SIZE
PATCH_SIZE = 16
DECODER_HIDDEN_DIM = 256
DROPOUT = 0.1               # Dropout probability

# Training Configuration
BATCH_SIZE = 8
LEARNING_RATE = 5e-4        # Balanced for 10 epochs
NUM_EPOCHS = 10             # Increased for full training
SUBSET_SIZE = None          # Use full dataset

# Continual Learning Configuration
NUM_TASKS = 5               # Test forgetting over multiple sequence steps
MEMORY_SIZE = 500           # Replay buffer size for ER/DER (approx 15-20% of data)
