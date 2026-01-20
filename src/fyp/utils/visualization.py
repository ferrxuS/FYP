import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to prevent hanging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from ..models.segmenter import ContainerDefectSegmenter
from ..data.dataset import ContainerDefectDataset
from ..config import SAVE_DIR

def plot_accuracy_matrix(results: dict, save_path: str = None):
    """Plot accuracy matrix as heatmap"""
    matrix = np.array(results['metrics']['Accuracy Matrix'])
    strategy_name = results.get('strategy_name', 'CL Strategy')
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        mask=mask,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'IoU'},
        square=True
    )

    plt.title(f'Accuracy Matrix - {strategy_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Task Evaluated On', fontsize=12)
    plt.ylabel('After Training Task', fontsize=12)

    # Set tick labels
    num_tasks = matrix.shape[0]
    tick_labels = [f'T{i+1}' for i in range(num_tasks)]
    plt.xticks(np.arange(num_tasks) + 0.5, tick_labels)
    plt.yticks(np.arange(num_tasks) + 0.5, tick_labels, rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # plt.show()

def plot_task_performance_curves(results: dict, save_path: str = None):
    """Plot how performance on each task changes as new tasks are learned"""
    matrix = np.array(results['metrics']['Accuracy Matrix'])
    strategy_name = results.get('strategy_name', 'CL Strategy')
    num_tasks = matrix.shape[0]

    plt.figure(figsize=(10, 6))

    for task_id in range(num_tasks):
        performance = []
        x_points = []

        for train_task in range(task_id, num_tasks):
            if matrix[train_task, task_id] > 0:
                performance.append(matrix[train_task, task_id])
                x_points.append(train_task + 1)

        plt.plot(
            x_points,
            performance,
            marker='o',
            linewidth=2,
            markersize=8,
            label=f'Task {task_id + 1}'
        )

    plt.xlabel('After Training Task', fontsize=12)
    plt.ylabel('IoU', fontsize=12)
    plt.title(f'Task Performance Over Time - {strategy_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, num_tasks + 1))
    plt.ylim([0, 1])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # plt.show()

def plot_strategy_comparison(strategy_results: Dict[str, dict], save_path: str = None):
    """Compare multiple CL strategies side-by-side"""
    strategies = []
    avg_accs = []
    forgettings = []
    bwts = []

    for name, results in strategy_results.items():
        if results is not None:
            strategies.append(name.upper())
            avg_accs.append(results['metrics']['Average Accuracy'])
            forgettings.append(results['metrics']['Forgetting'])
            bwts.append(results['metrics']['Backward Transfer'])

    if not strategies:
        print("No results to compare.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(strategies, avg_accs, color='steelblue', alpha=0.8)
    axes[0].set_ylabel('Average Accuracy', fontsize=12)
    axes[0].set_title('Average Accuracy\n(Higher is Better)', fontsize=12, fontweight='bold')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)

    axes[1].bar(strategies, forgettings, color='coral', alpha=0.8)
    axes[1].set_ylabel('Forgetting', fontsize=12)
    axes[1].set_title('Forgetting\n(Lower is Better)', fontsize=12, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)

    colors = ['green' if bwt >= 0 else 'red' for bwt in bwts]
    axes[2].bar(strategies, bwts, color=colors, alpha=0.8)
    axes[2].set_ylabel('Backward Transfer', fontsize=12)
    axes[2].set_title('Backward Transfer\n(Higher is Better)', fontsize=12, fontweight='bold')
    axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[2].grid(axis='y', alpha=0.3)

    for ax in axes:
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha='right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # plt.show()

def visualize_predictions(
    model: ContainerDefectSegmenter,
    dataset: ContainerDefectDataset,
    num_samples: int = 5,
    save_path: str = None
):
    """Visualize model predictions vs ground truth"""
    model.eval()
    device = next(model.parameters()).device
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask, filename = dataset[idx]

            image_batch = image.unsqueeze(0).to(device)
            prediction = model.predict(image_batch).cpu().squeeze()

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_denorm = image * std + mean
            image_denorm = torch.clamp(image_denorm, 0, 1)

            axes[i, 0].imshow(image_denorm.permute(1, 2, 0))
            axes[i, 0].set_title(f'Image\n{filename[:30]}...', fontsize=10)
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask.squeeze(), cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title('Ground Truth', fontsize=10)
            axes[i, 1].axis('off')

            axes[i, 2].imshow(prediction, cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title('Prediction', fontsize=10)
            axes[i, 2].axis('off')

            overlay = image_denorm.permute(1, 2, 0).clone()
            pred_mask = prediction > 0.5
            overlay[pred_mask, 0] = 1.0
            overlay[pred_mask, 1] = 0.0
            overlay[pred_mask, 2] = 0.0

            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay (Red = Defect)', fontsize=10)
            axes[i, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # plt.show()

def generate_cl_report(strategy_results: Dict[str, dict], save_dir: str = SAVE_DIR):
    """Generate a complete analysis report with all visualizations"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_strategy_comparison(strategy_results, save_path=save_dir / "cl_comparison.png")

    for strategy_name, results in strategy_results.items():
        if results is not None:
            plot_accuracy_matrix(results, save_path=save_dir / f"cl_matrix_{strategy_name}.png")
            plot_task_performance_curves(results, save_path=save_dir / f"cl_curves_{strategy_name}.png")

    print("Report generation complete!")
    print(f"Figures saved to: {save_dir}")
