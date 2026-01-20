from tqdm import tqdm
import torch
from .base import BaseCLStrategy
from ...training.metrics import compute_iou
from ...config import NUM_EPOCHS, DEVICE

class NaiveFineTuning(BaseCLStrategy):
    """Baseline: Just train sequentially without any CL strategy"""

    def __init__(self, model, device=DEVICE, lr=1e-3):
        super().__init__(model, device, lr)
        self.name = "Naive Fine-tuning"

    def train_task(self, task_loader, task_id, num_epochs=NUM_EPOCHS):
        self.model.train()
        print(f"Training Task {task_id+1} - {self.name}")

        avg_loss = 0.0
        avg_iou = 0.0

        for epoch in range(num_epochs):
            total_loss, total_iou, num_batches = 0.0, 0.0, 0

            for images, masks, _ in tqdm(task_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                images = images.to(self.device)
                targets = masks.squeeze(1).long().to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()

                predictions = torch.argmax(logits, dim=1)
                _, iou = compute_iou(predictions, targets)

                total_loss += loss.item()
                total_iou += iou
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_iou = total_iou / num_batches
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, IoU={avg_iou:.4f}")
        return {'loss': avg_loss, 'iou': avg_iou}

    def after_task(self, task_loader, task_id):
        pass
