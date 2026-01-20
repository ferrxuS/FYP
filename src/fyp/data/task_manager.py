import random
import torch
from torch.utils.data import Subset, DataLoader
from .dataset import ContainerDefectDataset
from ..config import NUM_TASKS, BATCH_SIZE

class TaskIncrementalDataset:
    """Splits dataset into sequential tasks for continual learning"""
    def __init__(
        self,
        base_dataset: ContainerDefectDataset,
        num_tasks: int = NUM_TASKS,
        shuffle_before_split: bool = True,
        seed: int = 42
    ):
        self.base_dataset = base_dataset
        self.num_tasks = num_tasks
        self.seed = seed

        total_size = len(base_dataset)
        indices = list(range(total_size))

        if shuffle_before_split:
            random.seed(seed)
            random.shuffle(indices)

        task_size = total_size // num_tasks
        self.task_indices = []

        for task_id in range(num_tasks):
            start_idx = task_id * task_size
            end_idx = total_size if task_id == num_tasks - 1 else (task_id + 1) * task_size
            self.task_indices.append(indices[start_idx:end_idx])
        
        print(f"Split {total_size} images into {num_tasks} tasks:")
        for i, task_idx in enumerate(self.task_indices):
            print(f"   Task{i+1}: {len(task_idx)} images")

    def get_task_dataset(self, task_id: int) -> Subset:
        assert 0 <= task_id < self.num_tasks
        return Subset(self.base_dataset, self.task_indices[task_id])

    def get_task_loader(
        self,
        task_id: int,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = True,
    ) -> DataLoader:
        task_dataset = self.get_task_dataset(task_id)
        return DataLoader(
            task_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )

    def get_all_tasks_until(self, task_id: int) -> Subset:
        combined_indices = []
        for i in range(task_id + 1):
            combined_indices.extend(self.task_indices[i])
        return Subset(self.base_dataset, combined_indices)
