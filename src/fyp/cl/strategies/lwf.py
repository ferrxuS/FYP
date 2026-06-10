import copy
from tqdm import tqdm
import torch
import torch.nn.functional as F
from .base import BaseCLStrategy
from ...training.metrics import compute_iou
from ...config import NUM_EPOCHS, DEVICE, LWF_ALPHA, LWF_TEMPERATURE

class LearningWithoutForgetting(BaseCLStrategy):
    """
    LwF: Learning without Forgetting — adapted from Mammoth for segmentation.
    Li & Hoiem, 2017.

    Distills knowledge from the frozen old model's pixel-wise logits
    instead of replaying stored samples. No buffer required.
    """

    def __init__(self, model, device=DEVICE, lr=1e-3,
                 alpha=LWF_ALPHA, temperature=LWF_TEMPERATURE):
        super().__init__(model, device, lr)
        self.name = "Learning Without Forgetting (LwF)"
        self.alpha = alpha
        self.temperature = temperature
        self.old_model = None

    def _distillation_loss(self, new_logits, old_logits):
        """
        Pixel-wise KL divergence between old and new model outputs.
        Operates on (B, C, H, W) segmentation logits.
        T² scaling follows Hinton et al. knowledge distillation convention.
        """
        assert new_logits.shape == old_logits.shape, (
            f"Shape mismatch: new={new_logits.shape}, old={old_logits.shape}"
        )
        T = self.temperature
        old_soft = F.softmax(old_logits / T, dim=1)
        new_logsoft = F.log_softmax(new_logits / T, dim=1)
        # Compute KL divergence per pixel (sum over channels), then average over batch and space
        kl_pixel = F.kl_div(new_logsoft, old_soft, reduction='none').sum(dim=1)
        loss = kl_pixel.mean() * (T ** 2)
        return loss

    def before_task(self, task_id):
        """
        Snapshot the current model as the frozen teacher before training
        the new task. Only fires from Task 2 onward.
        """
        if task_id > 0:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            for p in self.old_model.parameters():
                p.requires_grad = False

    def train_task(self, task_loader, task_id, num_epochs=NUM_EPOCHS):
        self.model.train()

        print(f"Training Task {task_id+1} - {self.name}")
        print(f"alpha={self.alpha}, temperature={self.temperature}")
        print(f"Distillation: {'ON' if self.old_model is not None else 'OFF (Task 1)'}")
        print(f"{'─'*60}")

        avg_loss = 0.0
        avg_iou = 0.0

        for epoch in range(num_epochs):
            total_loss, total_iou, num_batches = 0.0, 0.0, 0

            for images, masks, filenames in tqdm(task_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                images = images.to(self.device)
                targets = masks.squeeze(1).long().to(self.device)

                self.optimizer.zero_grad()
                new_logits = self.model(images)

                loss = self.criterion(new_logits, targets)

                if self.old_model is not None:
                    with torch.no_grad():
                        old_logits = self.old_model(images)
                    loss += self.alpha * self._distillation_loss(new_logits, old_logits)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                with torch.no_grad():
                    preds = torch.argmax(new_logits, dim=1)
                    _, iou = compute_iou(preds, targets)

                total_loss += loss.item()
                total_iou += iou
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_iou = total_iou / num_batches
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, IoU={avg_iou:.4f}")

        return {'loss': avg_loss, 'iou': avg_iou}

    def after_task(self, task_loader, task_id):
        pass
