import torch
import json
from pathlib import Path
from .config import (
    TRAIN_DIR, VALID_DIR, NUM_TASKS, NUM_EPOCHS, 
    SUBSET_SIZE, BACKBONE, NUM_CLASSES, DECODER_HIDDEN_DIM, 
    LEARNING_RATE, SAVE_DIR, RESULTS_DIR, DEVICE
)
from .data.dataset import ContainerDefectDataset
from .data.task_manager import TaskIncrementalDataset
from .models.segmenter import ContainerDefectSegmenter
from .cl.factory import get_cl_strategy
from .cl.trainer import CLTrainer
from .training.trainer_baseline import train_baseline_model
from .utils.visualization import generate_cl_report
from .utils.helpers import extract_dataset
from .config import ZIP_PATH, DATASET_ROOT

def run_cl_experiment(
    strategy_name: str,
    num_epochs_per_task: int = NUM_EPOCHS,
    subset_size: int = SUBSET_SIZE,
    save_results: bool = True
):
    """Run a complete CL experiment with a single strategy"""

    # Create datasets and task splits
    train_dataset = ContainerDefectDataset(TRAIN_DIR, subset_size=subset_size)
    val_dataset = ContainerDefectDataset(
        VALID_DIR,
        subset_size=subset_size // 2 if subset_size else None
    )

    train_tasks = TaskIncrementalDataset(train_dataset, num_tasks=NUM_TASKS)
    val_tasks = TaskIncrementalDataset(val_dataset, num_tasks=NUM_TASKS)

    # Create fresh model for this experiment
    model = ContainerDefectSegmenter(
        backbone_name=BACKBONE,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
        decoder_hidden_dim=DECODER_HIDDEN_DIM
    ).to(DEVICE)

    # Create CL strategy
    strategy = get_cl_strategy(strategy_name, model, DEVICE, lr=LEARNING_RATE)

    # Create trainer
    trainer = CLTrainer(
        strategy=strategy,
        task_manager=train_tasks,
        val_task_manager=val_tasks,
        device=DEVICE
    )

    # Run training
    results = trainer.train(num_epochs_per_task=num_epochs_per_task, save_dir=SAVE_DIR)

    # Save results
    if save_results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results_file = RESULTS_DIR / f"cl_results_{strategy_name}.json"
        trainer.save_results(results_file)

    return results

def run_all_cl_strategies(
    num_epochs_per_task: int = NUM_EPOCHS,
    subset_size: int = SUBSET_SIZE
):
    """Run all CL strategies and compare results"""
    strategies = ['naive', 'er', 'der++']
    all_results = {}

    print("\nRUNNING ALL CL STRATEGIES\n")
    print(f"Strategies to test: {strategies}")
    print(f"Epochs per task: {num_epochs_per_task}")
    print(f"Dataset size: {subset_size if subset_size else 'FULL'}")

    for strategy_name in strategies:
        try:
            results = run_cl_experiment(
                strategy_name=strategy_name,
                num_epochs_per_task=num_epochs_per_task,
                subset_size=subset_size,
                save_results=True
            )
            all_results[strategy_name] = results
        except Exception as e:
            print(f"\nError with {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[strategy_name] = None

    _print_comparison_summary(all_results)
    generate_cl_report(all_results, save_dir=RESULTS_DIR)
    return all_results

def _print_comparison_summary(all_results: dict):
    print("\nCOMPARISON SUMMARY\n")
    print(f"{'Strategy':<20} {'Avg Acc':>10} {'Forgetting':>12} {'BWT':>10}")
    print("-" * 60)

    for strategy_name, results in all_results.items():
        if results is not None:
            metrics = results['metrics']
            avg_acc = metrics['Average Accuracy']
            forgetting = metrics['Forgetting']
            bwt = metrics['Backward Transfer']
            print(f"{strategy_name.upper():<20} {avg_acc:>10.4f} {forgetting:>12.4f} {bwt:>10.4f}")
        else:
            print(f"{strategy_name.upper():<20} {'ERROR':>10} {'ERROR':>12} {'ERROR':>10}")

if __name__ == "__main__":
    # Extract dataset if missing
    extract_dataset(ZIP_PATH, DATASET_ROOT)
    
    # Train baseline model first
    print("\nTRAINING BASELINE MODEL\n")
    train_baseline_model(
        num_epochs=1,
        subset_size=10,
        save_path=SAVE_DIR / "baseline_model.pth"
    )

    # Run CL experiments
    run_all_cl_strategies(num_epochs_per_task=1, subset_size=10)
