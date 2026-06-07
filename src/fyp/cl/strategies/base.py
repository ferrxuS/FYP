import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from ...models.segmenter import ContainerDefectSegmenter
from ...training.metrics import compute_iou, compute_pixel_accuracy
from ...config import LEARNING_RATE, NUM_EPOCHS, DEVICE

class BaseCLStrategy(ABC):

    def __init__(
        self,
        model: ContainerDefectSegmenter,
        device: str = DEVICE,
        lr: float = LEARNING_RATE,
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        # Split learning rates: backbone parameters at 0.1x of head learning rate (lr)
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': lr * 0.1})
        if head_params:
            param_groups.append({'params': head_params, 'lr': lr})

        if param_groups:
            self.optimizer = optim.Adam(param_groups)
        else:
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr
            )

    @abstractmethod
    def train_task(
        self,
        task_loader: DataLoader,
        task_id: int,
        num_epochs: int = NUM_EPOCHS
    ):
        pass

    def before_task(self, task_id: int):
        pass

    @abstractmethod
    def after_task(self, task_loader: DataLoader, task_id: int):
        pass

    def evaluate_on_task(self, task_loader: DataLoader):
        self.model.eval()
        total_iou, total_acc, num_batches = 0.0, 0.0, 0

        with torch.no_grad():
            for images, masks, _ in task_loader:
                images = images.to(self.device)
                targets = masks.squeeze(1).long().to(self.device)

                logits = self.model(images)
                predictions = torch.argmax(logits, dim=1)

                _, iou = compute_iou(predictions, targets)
                acc = compute_pixel_accuracy(predictions, targets)

                total_iou += iou
                total_acc += acc
                num_batches += 1

        return total_iou / num_batches if num_batches > 0 else 0.0, total_acc / num_batches if num_batches > 0 else 0.0
