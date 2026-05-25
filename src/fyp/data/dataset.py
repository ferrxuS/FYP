import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import torchvision.transforms as T
from typing import Tuple, Optional, List, Dict
from pathlib import Path
from ..config import IMAGE_SIZE, PATCH_SIZE, DATASET_TYPE

class ContainerDefectDataset(Dataset):
    """Dataset for shipping container defect segmentation (supports both Old and New formats)"""

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = IMAGE_SIZE,
        patch_size: int = PATCH_SIZE,
        subset_size: Optional[int] = None
    ):
        self.root_dir = Path(root_dir)
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.image_size = image_size
        self.patch_size = patch_size
        self.dataset_type = DATASET_TYPE

        # Data collection
        self.image_files = []
        self.coco_data = None
        self.image_id_to_annotations = {}

        if self.dataset_type == "OLD":
            self._init_old_format(subset_size)
        else:
            self._init_coco_format(subset_size)

        # Image transform: resize + normalize
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        # Mask transform for OLD format: resize + scale [0,1]->[0,1]
        self.mask_transform = T.Compose([
            T.Resize(
                (image_size, image_size),
                interpolation=T.InterpolationMode.NEAREST
            ),
            T.ToTensor(),
            T.Lambda(lambda x: x * 255.0), # masks stored as [0,1] not [0,255]
        ])

        print(f"Loaded {len(self.image_files)} samples from {self.root_dir} ({self.dataset_type} mode)")

    def _init_old_format(self, subset_size):
        if not self.root_dir.exists():
            print(f"Warning: Directory {self.root_dir} does not exist.")
            return
        
        all_files = sorted(os.listdir(self.root_dir))
        self.image_files = [f for f in all_files if f.lower().endswith(".jpg")]
        
        if subset_size is not None:
            self.image_files = self.image_files[:subset_size]
        
        self._verify_masks()

    def _init_coco_format(self, subset_size):
        anno_path = self.root_dir / "_annotations.coco.json"
        if not anno_path.exists():
            # Try searching in subdirectories if extracted differently
            found_annos = list(self.root_dir.glob("**/_annotations.coco.json"))
            if found_annos:
                anno_path = found_annos[0]
            else:
                print(f"Warning: COCO annotations not found at {anno_path}")
                return

        with open(anno_path, 'r') as f:
            self.coco_data = json.load(f)

        # Map image_id to annotations
        from collections import defaultdict
        self.image_id_to_annotations = defaultdict(list)
        for anno in self.coco_data['annotations']:
            self.image_id_to_annotations[anno['image_id']].append(anno)

        # Filter images that actually exist and have annotations (optional)
        all_images = self.coco_data['images']
        if subset_size is not None:
            all_images = all_images[:subset_size]

        self.image_files = all_images # List of dicts in COCO mode

    def _verify_masks(self):
        """Verify all masks exist (for OLD format)"""
        missing = []
        for img in self.image_files:
            base = os.path.splitext(img)[0]
            mask_path = self.root_dir / f"{base}_mask.png"
            if not mask_path.exists():
                missing.append(str(mask_path))
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} masks")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        if self.dataset_type == "OLD":
            return self._get_old_item(idx)
        else:
            return self._get_coco_item(idx)

    def _get_old_item(self, idx):
        img_name = self.image_files[idx]
        img_path = self.root_dir / img_name
        mask_path = self.root_dir / f"{os.path.splitext(img_name)[0]}_mask.png"

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float() # binarize (Background=0, Defect=1)

        return image, mask, img_name

    def _get_coco_item(self, idx):
        img_info = self.image_files[idx]
        img_id = img_info['id']
        img_name = img_info['file_name']
        
        # Paths might be relative to JSON or root
        img_path = self.root_dir / img_name
        if not img_path.exists():
            # Handle potential nested folder structure
            img_path = Path(next(self.root_dir.glob(f"**/{img_name}")))

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        # Create multi-class mask
        # 0 is Background, 1-6 are categories
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        
        annotations = self.image_id_to_annotations.get(img_id, [])
        for anno in annotations:
            cat_id = anno['category_id']
            # Roboflow category IDs 1-5 map to labels 1-5
            # Background is 0
            label = cat_id
            
            segmentation = anno['segmentation']
            if isinstance(segmentation, list):
                for poly in segmentation:
                    if len(poly) >= 4: # Need at least 2 points
                        draw.polygon(poly, fill=label)
            # RLE not handled here for simplicity, but could be added with pycocotools

        image = self.image_transform(image)
        # Resize mask with NEAREST to preserve class labels
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).float()
        
        return image, mask_tensor, img_name

    def patch_grid(self) -> Tuple[int, int]:
        """Returns (H_patches, W_patches)"""
        n = self.image_size // self.patch_size
        return n, n
