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
        mode: str = "class", # "random", "sequential", or "class"
        task_keywords: list = None,
        seed: int = 42
    ):
        self.base_dataset = base_dataset
        self.num_tasks = num_tasks
        self.seed = seed
        
        if mode == "class":
            self._split_by_class(task_keywords)
        else:
            self._split_by_index(mode == "random")

    def _split_by_index(self, shuffle: bool):
        total_size = len(self.base_dataset)
        indices = list(range(total_size))

        if shuffle:
            random.seed(self.seed)
            random.shuffle(indices)

        task_size = total_size // self.num_tasks
        self.task_indices = []

        for task_id in range(self.num_tasks):
            start_idx = task_id * task_size
            end_idx = total_size if task_id == self.num_tasks - 1 else (task_id + 1) * task_size
            self.task_indices.append(indices[start_idx:end_idx])

    def _split_by_class(self, task_keywords: list):
        """Splits data by starting characters or keyword matches in filenames"""
        if task_keywords is None:
            # Grouping by starting characters to create balanced source-shifts
            task_keywords = [
                ["C"],              # Task 1: Containers starting with C (e.g. CSNU, CSLU)
                ["T"],              # Task 2: Containers starting with T (e.g. TCNU, TRHU)
                ["F", "B", "A", "D"], # Task 3: F, B, A, D prefixes
                ["S", "W", "G", "M", "R", "O", "U", "E", "K", "P", "H", "Y", "N", "L"], # Task 4: Other prefixes
                ["standard_id"]     # Task 5: Everything else (primarily IMG... files)
            ]
        
        self.num_tasks = len(task_keywords)
        self.task_indices = [[] for _ in range(self.num_tasks)]
        
        all_files = self.base_dataset.image_files
        used_indices = set()

        for t_idx, prefixes in enumerate(task_keywords):
            # Special case for the last task to catch-all
            if "standard_id" in prefixes:
                for f_idx in range(len(all_files)):
                    if f_idx not in used_indices:
                        self.task_indices[t_idx].append(f_idx)
                        used_indices.add(f_idx)
                continue

            for f_idx, filename in enumerate(all_files):
                if f_idx in used_indices:
                    continue
                
                # Match by starting character (prefix)
                if any(filename.upper().startswith(p.upper()) for p in prefixes):
                    self.task_indices[t_idx].append(f_idx)
                    used_indices.add(f_idx)

        print(f"\nClass/Source-Incremental Split ({len(all_files)} images):")
        for i, task_idx in enumerate(self.task_indices):
            print(f"   Task{i+1}: {len(task_idx)} images (Prefixes: {task_keywords[i]})")

    def get_task_dataset(self, task_id: int) -> Subset:
        assert 0 <= task_id < self.num_tasks
        if len(self.task_indices[task_id]) == 0:
            print(f"Warning: Task {task_id+1} is empty!")
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
