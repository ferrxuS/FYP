import torch
import json
from .strategies.base import BaseCLStrategy
from ..data.task_manager import TaskIncrementalDataset
from .metrics import CLMetricsTracker
from ..config import NUM_EPOCHS, SAVE_DIR, BATCH_SIZE, DEVICE

class CLTrainer:
    """Orchestrates continual learning training and evaluation"""
    def __init__(
        self,
        strategy: BaseCLStrategy,
        task_manager: TaskIncrementalDataset,
        val_task_manager: TaskIncrementalDataset = None,
        device: str = DEVICE
    ):
        self.strategy = strategy
        self.task_manager = task_manager
        self.val_task_manager = val_task_manager
        self.device = device
        self.num_tasks = task_manager.num_tasks
        self.metrics = CLMetricsTracker(num_tasks=self.num_tasks)
        self.pixel_metrics = CLMetricsTracker(num_tasks=self.num_tasks)
        self.training_history = []

    def train(self, num_epochs_per_task: int = NUM_EPOCHS, save_dir: str = SAVE_DIR):

        print(f"\nCONTINUAL LEARNING TRAINING: {self.strategy.name}\n")
        print(f"Tasks: {self.num_tasks}")
        print(f"Epochs per task: {num_epochs_per_task}")
        print(f"Device: {self.device}\n")

        for task_id in range(self.num_tasks):
            print(f"\n{'█'*60}")
            print(f"TASK {task_id + 1}/{self.num_tasks}")
            print(f"{'█'*60}")

            task_loader = self.task_manager.get_task_loader(task_id, batch_size=BATCH_SIZE, shuffle=True)
            train_metrics = self.strategy.train_task(task_loader, task_id, num_epochs=num_epochs_per_task)
            self.strategy.after_task(task_loader, task_id)

            print(f"\n  Evaluating on all tasks...")
            task_ious, task_accs = self._evaluate_all_tasks(task_id)
            self.metrics.update(task_id, task_ious)
            self.pixel_metrics.update(task_id, task_accs)
            self._print_task_results(task_id, task_ious, task_accs)

            # Save checkpoint
            checkpoint_path = save_dir / f"{self.strategy.name.replace(' ', '_')}_task{task_id+1}.pth"
            torch.save(self.strategy.model.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

            self.training_history.append({
                'task_id': task_id,
                'train_metrics': train_metrics,
                'task_accuracies': task_ious,
                'task_pixel_accuracies': task_accs,
                'avg_accuracy': self.metrics.compute_average_accuracy(task_id),
                'avg_pixel_accuracy': self.pixel_metrics.compute_average_accuracy(task_id),
                'forgetting': self.metrics.compute_forgetting(task_id),
                'pixel_forgetting': self.pixel_metrics.compute_forgetting(task_id)
            })

        print("TRAINING COMPLETE")
        print("\n" + "="*60 + "\nSUMMARY OF IoU METRICS\n" + "="*60)
        self.metrics.print_summary()
        
        print("\n" + "="*60 + "\nSUMMARY OF PIXEL ACCURACY METRICS\n" + "="*60)
        self.pixel_metrics.print_summary()

        return {
            'metrics': self.metrics.get_summary(),
            'pixel_metrics': self.pixel_metrics.get_summary(),
            'history': self.training_history,
            'strategy_name': self.strategy.name
        }

    def _evaluate_all_tasks(self, current_task: int):
        task_ious = {}
        task_accs = {}

        for eval_task_id in range(current_task + 1):
            if self.val_task_manager is not None:
                eval_loader = self.val_task_manager.get_task_loader(eval_task_id, batch_size=BATCH_SIZE, shuffle=False)
            else:
                eval_loader = self.task_manager.get_task_loader(eval_task_id, batch_size=BATCH_SIZE, shuffle=False)

            iou, acc = self.strategy.evaluate_on_task(eval_loader)
            task_ious[eval_task_id] = iou
            task_accs[eval_task_id] = acc

        return task_ious, task_accs

    def _print_task_results(self, current_task: int, task_ious: dict, task_accs: dict):
        print(f"\n  Performance after Task {current_task + 1}:")
        for task_id in task_ious.keys():
            iou = task_ious[task_id]
            acc = task_accs[task_id]
            print(f"    Task {task_id + 1}: IoU = {iou:.4f} | Pixel Acc = {acc:.4f}")

        avg_iou = self.metrics.compute_average_accuracy(current_task)
        forgetting_iou = self.metrics.compute_forgetting(current_task)

        avg_acc = self.pixel_metrics.compute_average_accuracy(current_task)
        forgetting_acc = self.pixel_metrics.compute_forgetting(current_task)

        print(f"  ─────────────────────────────")
        print(f"    Avg IoU:      {avg_iou:.4f} | Avg Pixel Acc: {avg_acc:.4f}")
        print(f"    Forgetting:   {forgetting_iou:.4f} | Forgetting Acc: {forgetting_acc:.4f}")

    def save_results(self, filepath: str):
        results = {
            'strategy': self.strategy.name,
            'num_tasks': self.num_tasks,
            'metrics': self.metrics.get_summary(),
            'pixel_metrics': self.pixel_metrics.get_summary(),
            'history': self.training_history
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {filepath}")
