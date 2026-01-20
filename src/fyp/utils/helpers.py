import os
import torch
import zipfile
from pathlib import Path
from ..models.segmenter import ContainerDefectSegmenter
from ..config import BACKBONE, NUM_CLASSES, DECODER_HIDDEN_DIM, SAVE_DIR

def extract_dataset(zip_path, extract_to):
    """Extract dataset zip file"""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    
    # Check if a marker file/folder exists to skip extraction
    if (extract_to / "train").exists() and (extract_to / "valid").exists():
        print("Dataset already extracted and valid directories found")
        return
        
    if not zip_path.exists():
        print(f"Zip file {zip_path} not found. Extraction skipped.")
        return

    print(f"Extracting dataset to {extract_to}...")
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            print("Dataset extracted successfully")
    except Exception as e:
        print(f"Error extracting dataset: {e}")

def load_trained_model(model_path, device='cuda'):
    """Load a saved model checkpoint"""
    model = ContainerDefectSegmenter(
        backbone_name=BACKBONE,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
        decoder_hidden_dim=DECODER_HIDDEN_DIM,
    )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    print(f"Model loaded from: {model_path}")
    return model
