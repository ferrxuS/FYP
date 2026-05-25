import random
import torch
from torch.utils.data import Subset, DataLoader, Dataset
from typing import List, Optional
from .dataset import ContainerDefectDataset
from ..config import NUM_TASKS, BATCH_SIZE, IGNORE_COMPOSITE, ALLOW_REUSE

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
        """
        Splits data into tasks.
        - For OLD dataset: Splits by filename prefix (Source-Incremental).
        - For NEW dataset: Splits by COCO category IDs (Class-Incremental).
        """
        all_files = self.base_dataset.image_files
        self.task_indices = []
        
        if self.base_dataset.dataset_type == "OLD":
            self._split_by_source_old(task_keywords)
        else:
            self._split_by_category_new(task_keywords)

    def _split_by_source_old(self, task_keywords: list):
        """Original source-incremental logic for binary-mask dataset"""
        if task_keywords is None:
            task_keywords = [
                ["C"], ["T"], ["F", "B", "A", "D"], 
                ["S", "W", "G", "M", "R", "O", "U", "E", "K", "P", "H", "Y", "N", "L"], 
                ["standard_id"]
            ]
        
        self.num_tasks = len(task_keywords)
        self.task_indices = [[] for _ in range(self.num_tasks)]
        all_files = self.base_dataset.image_files
        used_indices = set()

        for t_idx, prefixes in enumerate(task_keywords):
            if "standard_id" in prefixes:
                for f_idx in range(len(all_files)):
                    if f_idx not in used_indices:
                        self.task_indices[t_idx].append(f_idx)
                        used_indices.add(f_idx)
                continue

            for f_idx, filename in enumerate(all_files):
                if f_idx in used_indices: continue
                if any(filename.upper().startswith(p.upper()) for p in prefixes):
                    self.task_indices[t_idx].append(f_idx)
                    used_indices.add(f_idx)

        print(f"\nSource-Incremental Split (OLD): {sum(len(t) for t in self.task_indices)} images")

    def _split_by_category_new(self, task_categories: list):
        """Class-incremental logic for COCO dataset"""
        if task_categories is None:
            # Grouping 6 active classes into 5 tasks
            # IDs in this dataset: 0: DEFECT, 1: PAINT, 2: PATCH, 3: REMOVE, 4: SEALANT, 5: WASH
            task_categories = [
                [1],    # Task 1: Paint
                [2],    # Task 2: Patch
                [3],    # Task 3: Remove
                [4],    # Task 4: Sealant
                [5]     # Task 5: Wash
            ]
        
        self.task_categories = task_categories # Store for label masking
        self.num_tasks = len(task_categories)
        self.task_indices = [[] for _ in range(self.num_tasks)]
        used_indices = set()
        img_to_cats = self.base_dataset.image_id_to_annotations
        
        for t_idx, cat_ids in enumerate(task_categories):
            for f_idx, img_info in enumerate(self.base_dataset.image_files):
                # 1. Handle image reuse
                if not ALLOW_REUSE and f_idx in used_indices: 
                    continue
                
                img_id = img_info['id']
                annos = img_to_cats.get(img_id, [])
                img_cats = set(a['category_id'] for a in annos)
                
                # 2. Handle composite images (multiple defect types)
                if IGNORE_COMPOSITE and len(img_cats) > 1:
                    continue
                
                # 3. Assign to task if it contains any of the target categories
                if any(c in cat_ids for c in img_cats):
                    # For safety, don't double-add the same image to the SAME task
                    if f_idx not in self.task_indices[t_idx]:
                        self.task_indices[t_idx].append(f_idx)
                        used_indices.add(f_idx)

        print(f"\nClass-Incremental Split (NEW): {sum(len(t) for t in self.task_indices)} images total")
        for i, t in enumerate(self.task_indices):
            print(f"   Task{i+1}: {len(t)} images (Categories: {task_categories[i]})")
            if len(t) == 0:
                print(f"   WARNING: Task{i+1} is EMPTY! Try increasing subset_size.")

    def get_task_dataset(self, task_id: int) -> Dataset:
        assert 0 <= task_id < self.num_tasks
        if len(self.task_indices[task_id]) == 0:
            print(f"Warning: Task {task_id+1} is empty!")
        
        subset = Subset(self.base_dataset, self.task_indices[task_id])
        
        # Apply label masking if using Class-Incremental
        if hasattr(self, 'task_categories'):
            allowed_cats = self.task_categories[task_id]
            return LabelMaskingDataset(subset, allowed_cats)
            
        return subset

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

class LabelMaskingDataset(Dataset):
    """
    Wrapper that masks out labels not belonging to the current task.
    Ensures 'future' classes are treated as background.
    """
    def __init__(self, dataset: Dataset, allowed_categories: List[int]):
        self.dataset = dataset
        # category_id X becomes label X in the mask. Background is 0.
        self.allowed_labels = allowed_categories + [0]
        self.allowed_labels_tensor = torch.tensor(self.allowed_labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, mask, filename = self.dataset[idx]
        
        # Create a mask of which pixels are allowed
        is_allowed = torch.isin(mask, self.allowed_labels_tensor)
        
        # Set everything else to 0 (Background)
        masked_mask = torch.where(is_allowed, mask, torch.zeros_like(mask))
        
        return image, masked_mask, filename
