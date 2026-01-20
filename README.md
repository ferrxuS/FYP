# Shipping Container Defect Segmentation using Continual Learning

This project implements a segmentation pipeline for detecting defects in shipping containers using DINOv3 backbones and various Continual Learning (CL) strategies.

## Project Structure

- `src/fyp/`: Main package containing the source code.
  - `config.py`: Configuration and hyperparameters.
  - `data/`: Data loading and task management.
  - `models/`: DINOv3 feature extractor and segmentation decoder.
  - `cl/`: Continual learning strategies (ER, DER++, etc.).
  - `training/`: Training utilities and metrics.
  - `utils/`: Visualization and helper functions.
- `notebooks/`: Jupyter notebooks for experimentation.
- `data/`: Directory for dataset files.
- `models/`: Directory for saved model checkpoints.
- `results/`: Directory for experiment results and plots.

## Setup

This project uses `uv` for package management.

### Installation

1. Install `uv`:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

### Running Experiments

You can run the full suite of CL experiments:
```bash
uv run python -m src.fyp.main
```

Or use the refactored notebook in `notebooks/FYP1_Refactored.ipynb`.

## Features

- **Backbone**: DINOv3 (ViT series)
- **CL Strategies**: Naive Fine-tuning, Experience Replay (ER), Dark Experience Replay (DER++)
- **Metrics**: Average Accuracy, Forgetting, Backward Transfer
- **Visualization**: Accuracy Matrix Heatmaps, Performance Curves, Strategy Comparison
# FYP
