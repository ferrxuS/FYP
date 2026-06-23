# Shipping Container Defect Segmentation via Continual Learning

A state-of-the-art Continual Learning (CL) framework for dense pixel-level shipping container defect segmentation. This project adapts advanced CL strategies (rehearsal, regularization, and meta-learning) from image classification to dense spatial prediction tasks. It utilizes self-supervised Vision Foundation Models (DINOv3) to address catastrophic forgetting.

---

## Key Highlights & Architecture

*   **Vision Foundation Model Backbone:** Powered by **DINOv3 (`vitl16` / ~305M parameters)**. Supports fully frozen feature extraction or selective unfreezing of the last $N$ Transformer blocks.
*   **Progressive Upsampling Decoder:** A convolutional decoder head that progressively upsamples ViT patch tokens ($48 \times 48$) to full target resolution ($768 \times 768$) via:
    $$\text{Conv2d(3x3)} \rightarrow \text{BatchNorm} \rightarrow \text{ReLU} \rightarrow \text{Dropout (p=0.1)} \rightarrow \text{Bilinear Upsampling (2x)}$$
*   **8 Continual Learning Strategies:** Re-engineered for spatial `(B, C, H, W)` inputs and dense pixel targets:
    *   *Regularization-based:* **Synaptic Intelligence (SI)**, **Learning Without Forgetting (LwF)**.
    *   *Rehearsal-based:* **Experience Replay (ER)**, **Experience Replay with Asymmetric Cross-Entropy (ER-ACE)**, **Dark Experience Replay++ (DER++)**.
    *   *Meta-Learning & Classifier Alignment:* **Meta-Experience Replay (MER)**, **Slow Learner Classifier Alignment (SLCA)**.
    *   *Baseline:* **Naive Fine-Tuning** (baseline experiencing high catastrophic forgetting).
*   **Multi-Class Incremental Datasets:** Supports a binary-mask dataset and a 6-class COCO-formatted dataset (Background + 5 defect types: paint repair, patch repair, removal repair, sealant repair, and wash repair).

---

## Directory Structure

```text
├── FYP2.ipynb                      # Synchronized Google Colab notebook (interactive training/evaluation)
├── pyproject.toml                  # Project metadata & Python packaging configs
├── uv.lock                         # Lockfile managed by uv
├── src/
│   └── fyp/
│       ├── config.py               # Core hyperparameters & project directory paths
│       ├── main.py                 # Primary CLI runner for single or multi-strategy experiments
│       ├── app.py                  # Streamlit Interactive Dashboard & Model Inference UI
│       ├── cl/
│       │   ├── buffer.py           # Reservoir-sampling experience replay buffer
│       │   ├── factory.py          # Dynamic factory instantiation for CL strategies
│       │   ├── metrics.py          # Trackers for Average Accuracy, Forgetting, and Backward Transfer
│       │   ├── trainer.py          # Orchestrates training loops over tasks
│       │   └── strategies/         # Strategy implementations (naive, er, derpp, lwf, si, slca, mer, er_ace)
│       ├── data/
│       │   ├── dataset.py          # Dual-format dataset loader (OLD/COCO) with polygon rasterization
│       │   └── task_manager.py     # Class-incremental task partitioner & label masking loader
│       ├── models/
│       │   ├── backbone.py         # DINOv3 feature extractor & selective unfreezer
│       │   ├── decoder.py          # Progressive upsampling convolutional segmentation head
│       │   └── segmenter.py        # Combined model module (Backbone + Head)
│       └── utils/
│           ├── helpers.py          # Checkpoint IO, path indexing, and device configurations
│           ├── visualization.py    # Metric heatmap generation & sample predictions plotting
│           └── generate_full_comparison.py # Comparative plotter for Frozen vs. Unfrozen backbone runs
```

---

## Setup & Installation

This project uses [**`uv`**](https://github.com/astral-sh/uv) for fast, reliable package management.

### **1. Install `uv`**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### **2. Synchronize Virtual Environment & Dependencies**
Navigate to the root directory and run:
```bash
uv sync
```
This automatically sets up a `.venv` virtual environment and installs all dependencies listed in `pyproject.toml` (including PyTorch, Streamlit, and Hugging Face Transformers).

---

## Running Experiments

### **Running Locally / Interactive Shell**
To run a series of Continual Learning strategies on your local environment:
```bash
# Run specified CL strategies (e.g., ER-ACE and DER++)
uv run python -m src.fyp.main --strategies er_ace der++

# Run the Joint baseline model (multi-task learning with all data present at once)
uv run python -m src.fyp.main --joint
```

---

## Evaluation & Comparative Visualization

The system generates comparative analytics including **Average Accuracy**, **Forgetting Rate**, and **Backward Transfer (BWT)** curves.

### **Generating Side-by-Side Plots (Frozen vs. Unfrozen Backbones)**
To generate grouped comparative bar charts between different experiment runs:
```bash
uv run python src/fyp/utils/generate_full_comparison.py --runs run_1 run_2 --output-name final_comparison.png
```

---

## Streamlit Interactive Dashboard

You can explore results, compare strategy benchmarks, and perform live defect inference using the Streamlit dashboard.

### **Launch Locally**
```bash
uv run streamlit run src/fyp/app.py --server.port 8501
```

Access the dashboard in your browser at `http://localhost:8501`.

---

## Continual Learning Strategy Adaptations

*   **Experience Replay (ER):** Concatenates current images and targets with replayed items into unified 4D tensors `(B + B_buf, 3, 768, 768)` to compute dense, pixel-wise cross-entropy.
*   **ER-ACE (Asymmetric Cross-Entropy):** Broadcasts asymmetric masks `(1, C, 1, 1)` across spatial logit channels to filter out future/unseen categories during forward passes. This prevents negative gradients from penalizing active pixels as background.
*   **Dark Experience Replay++ (DER++):** Computes pixel-wise distillation Mean Squared Error (MSE) between the active model output and dense 4D logit maps stored in the replay buffer. To manage memory, logit tensors are moved to the CPU (`.cpu()`) post-forward pass.
*   **Synaptic Intelligence (SI):** Track weights parameter-by-parameter in a Python dictionary (`self.big_omega`) strictly for trainable parameters (the decoder head). This avoids allocating flat tensors for the massive DINOv3 backbone.
*   **Meta-Experience Replay (MER):** Computes Reptile inner-loop ($\beta$) and outer-loop ($\gamma$) meta-gradients strictly within the decoder's parameter space.
*   **Slow Learner Classifier Alignment (SLCA):** Resizes mask coordinates onto the $48 \times 48$ feature token grid, extracts token vectors matching active defect labels to compute class means, and trains the decoder on these spatial prototypes.
*   **Learning Without Forgetting (LwF):** Computes pixel-level Kullback-Leibler (KL) divergence between current logit maps and a pre-task snapshot model, scaling distillation by the square of temperature ($T^2$). This is the only strategy that runs fully buffer-free.
