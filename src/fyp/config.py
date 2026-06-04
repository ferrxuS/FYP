import torch
from pathlib import Path

# Dataset Selection
DATASET_TYPE = "NEW"  # Options: "OLD" (binary masks), "NEW" (COCO Segmentation)

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Project Roots
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Old Dataset Paths (binary-mask dataset)
OLD_DATASET_ROOT = DATA_DIR / "dataset"
OLD_ZIP_PATH = DATA_DIR / "binary_mask_dataset.zip"

# New Dataset Paths (COCO Segmentation)
NEW_DATASET_ROOT = DATA_DIR / "coco_dataset"
NEW_ZIP_PATH = DATA_DIR / "coco_dataset.zip"

# Active Dataset Selection
if DATASET_TYPE == "OLD":
    DATASET_ROOT = OLD_DATASET_ROOT
    NUM_CLASSES = 2             # Binary: background + defect
    ZIP_PATH = OLD_ZIP_PATH
else:
    DATASET_ROOT = NEW_DATASET_ROOT
    NUM_CLASSES = 6             # Background + 5 active COCO classes
    ZIP_PATH = NEW_ZIP_PATH

TRAIN_DIR = DATASET_ROOT / "train"
VALID_DIR = DATASET_ROOT / "valid"
SAVE_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Model Configuration
BACKBONE = "vitl16"         # DINOv3 variant: vits16, vitb16, vitl16, vith16
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
MER_BETA = 0.1              # within-batch Reptile LR
MER_GAMMA = 0.1             # across-batch Reptile LR
MER_BATCH_NUM = 1           # number of inner iterations per observe call

# SI Hyperparameters (aimagelab/mammoth architecture)
SI_C = 0.5                  # penalty weight - higher = more protection, less plasticity
SI_XI = 1.0                 # damping term - prevents division by zero

# LwF Hyperparameters
LWF_ALPHA = 0.5             # distillation loss weight
LWF_TEMPERATURE = 2.0       # softmax temperature

# SLCA Hyperparameters
SLCA_CA_EPOCHS = 5          # alignment epochs
SLCA_LOGIT_NORM = 0.1       # logit L2 norm weight

# Task Splitting Options
IGNORE_COMPOSITE = False    # If True, skip images that have more than one type of defect
ALLOW_REUSE = False         # If True, images with multiple defects can appear in multiple tasks
