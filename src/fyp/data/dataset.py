import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import Tuple, Optional, List
from pathlib import Path
from ..config import IMAGE_SIZE, PATCH_SIZE

class ContainerDefectDataset(Dataset):
    """Dataset for shipping container defect segmentation"""

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

        # Collect image files
        if not self.root_dir.exists():
            print(f"Warning: Directory {self.root_dir} does not exist.")
            self.image_files = []
        else:
            all_files = sorted(os.listdir(self.root_dir))
            self.image_files: List[str] = [
                f for f in all_files if f.lower().endswith(".jpg")
            ]

        if subset_size is not None:
            self.image_files = self.image_files[:subset_size]

        if self.image_files:
            self._verify_masks()

        # Image transform: resize + normalize
        self.image_transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        # Mask transform: resize + scale [0,1]->[0,1]
        self.mask_transform = T.Compose([
            T.Resize(
                (image_size, image_size),
                interpolation=T.InterpolationMode.NEAREST
            ),
            T.ToTensor(),
            T.Lambda(lambda x: x * 255.0), # masks stored as [0,1] not [0,255]
        ])

        print(f"Loaded {len(self.image_files)} samples from {self.root_dir}")

    def _verify_masks(self):
        """Verify all masks exist"""
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
        img_name = self.image_files[idx]
        base = os.path.splitext(img_name)[0]
        mask_name = f"{base}_mask.png"

        img_path = self.root_dir / img_name
        mask_path = self.root_dir / mask_name

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float() # binarize

        return image, mask, img_name

    def patch_grid(self) -> Tuple[int, int]:
        """Returns (H_patches, W_patches)"""
        n = self.image_size // self.patch_size
        return n, n
