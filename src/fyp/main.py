import torch
import json
from pathlib import Path
from .config import (
    TRAIN_DIR, VALID_DIR, NUM_TASKS, NUM_EPOCHS, 
    SUBSET_SIZE, BACKBONE, NUM_CLASSES, DECODER_HIDDEN_DIM, 
    LEARNING_RATE, SAVE_DIR, RESULTS_DIR, DEVICE, UNFREEZE_LAST_N_BLOCKS
)
from .data.dataset import ContainerDefectDataset
from .data.task_manager import TaskIncrementalDataset
from .models.segmenter import ContainerDefectSegmenter
from .cl.factory import get_cl_strategy
from .cl.trainer import CLTrainer
from .training.trainer_baseline import train_baseline_model
from .utils.visualization import generate_cl_report
from .utils.helpers import extract_dataset, get_next_run_index
from .config import ZIP_PATH, DATASET_ROOT

def run_cl_experiment(
    strategy_name: str,
    num_epochs_per_task: int = NUM_EPOCHS,
    subset_size: int = SUBSET_SIZE,
    save_results: bool = True,
    save_dir: Path = SAVE_DIR,
    results_dir: Path = RESULTS_DIR,
    unfreeze_last_n_blocks: int = UNFREEZE_LAST_N_BLOCKS
):
    """Run a complete CL experiment with a single strategy"""

    # Create datasets and task splits
    train_dataset = ContainerDefectDataset(TRAIN_DIR, subset_size=subset_size)
    val_dataset = ContainerDefectDataset(
        VALID_DIR,
        subset_size=subset_size // 2 if subset_size else None
    )

    train_tasks = TaskIncrementalDataset(train_dataset, num_tasks=NUM_TASKS, mode="class")
    val_tasks = TaskIncrementalDataset(val_dataset, num_tasks=NUM_TASKS, mode="class")

    # Create fresh model for this experiment
    model = ContainerDefectSegmenter(
        backbone_name=BACKBONE,
        num_classes=NUM_CLASSES,
        freeze_backbone=True,
        unfreeze_last_n_blocks=unfreeze_last_n_blocks,
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
    results = trainer.train(num_epochs_per_task=num_epochs_per_task, save_dir=save_dir)

    # Save results
    if save_results:
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"cl_results_{strategy_name}.json"
        trainer.save_results(results_file)

    return results

def run_all_cl_strategies(
    num_epochs_per_task: int = NUM_EPOCHS,
    subset_size: int = SUBSET_SIZE,
    save_dir: Path = SAVE_DIR,
    results_dir: Path = RESULTS_DIR,
    strategies: list = None,
    unfreeze_last_n_blocks: int = UNFREEZE_LAST_N_BLOCKS
):
    """Run all CL strategies and compare results"""
    if strategies is None:
        strategies = ['der++']
    all_results = {}

    print("\nRUNNING ALL CL STRATEGIES\n")
    print(f"Strategies to test: {strategies}")
    print(f"Epochs per task: {num_epochs_per_task}")
    print(f"Dataset size: {subset_size if subset_size else 'FULL'}")
    print(f"Unfreeze last N blocks: {unfreeze_last_n_blocks}")

    for strategy_name in strategies:
        try:
            results = run_cl_experiment(
                strategy_name=strategy_name,
                num_epochs_per_task=num_epochs_per_task,
                subset_size=subset_size,
                save_results=True,
                save_dir=save_dir,
                results_dir=results_dir,
                unfreeze_last_n_blocks=unfreeze_last_n_blocks
            )
            all_results[strategy_name] = results
        except Exception as e:
            print(f"\nError with {strategy_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[strategy_name] = None

    _print_comparison_summary(all_results)
    generate_cl_report(all_results, save_dir=results_dir)
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
    import argparse
    parser = argparse.ArgumentParser(description="Continual Learning Experiment Runner")
    parser.add_argument(
        "--strategies",
        nargs="+",
        type=str,
        default=["er_ace"],
        help="List of strategies to run (e.g., naive er der++ lwf slca er_ace)"
    )
    parser.add_argument(
        "--unfreeze_last_n_blocks",
        type=int,
        default=UNFREEZE_LAST_N_BLOCKS,
        help="Number of last blocks to unfreeze in backbone"
    )
    args = parser.parse_args()

    # Extract dataset if missing
    extract_dataset(ZIP_PATH, DATASET_ROOT)

    # Get next available run index (synced for both models and results)
    run_idx = get_next_run_index([SAVE_DIR, RESULTS_DIR])
    current_save_dir = SAVE_DIR / f"run_{run_idx}"
    current_results_dir = RESULTS_DIR / f"run_{run_idx}"
    
    # Create directories immediately
    current_save_dir.mkdir(parents=True, exist_ok=True)
    current_results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Experiment Run: {run_idx}")
    print(f"Output Directory (Models): {current_save_dir}")
    print(f"Output Directory (Results): {current_results_dir}")

    # Train baseline model first (uncomment to run baseline)
    # print("\nTRAINING BASELINE MODEL\n")
    # train_baseline_model(
    #     num_epochs=NUM_EPOCHS,
    #     subset_size=SUBSET_SIZE,
    #     save_path=current_save_dir / "baseline_model.pth"
    # )

    # Run CL experiments
    run_all_cl_strategies(
        num_epochs_per_task=NUM_EPOCHS,
        subset_size=SUBSET_SIZE,
        save_dir=current_save_dir,
        results_dir=current_results_dir,
        strategies=args.strategies,
        unfreeze_last_n_blocks=args.unfreeze_last_n_blocks
    )
