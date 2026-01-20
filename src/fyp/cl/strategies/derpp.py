from tqdm import tqdm
import torch
import torch.nn as nn
import random
from .base import BaseCLStrategy
from ...training.metrics import compute_iou
from ...config import NUM_EPOCHS, MEMORY_SIZE, DEVICE

class DarkExperienceReplayPP(BaseCLStrategy):
    """DER++: Experience Replay + Knowledge Distillation"""

    def __init__(self, model, device=DEVICE, lr=1e-3,
                 buffer_size=MEMORY_SIZE, samples_per_task=100,
                 alpha=0.5, beta=0.5):
        super().__init__(model, device, lr)
        self.name = "DER++"
        self.buffer = []
        self.buffer_size = buffer_size
        self.samples_per_task = samples_per_task
        self.alpha = alpha
        self.beta = beta

    def train_task(self, task_loader, task_id, num_epochs=NUM_EPOCHS):
        """Train with replay + logit distillation."""
        self.model.train()

        print(f"Training Task {task_id+1} - {self.name}")
        print(f"Buffer: {len(self.buffer)}/{self.buffer_size} samples")
        print(f"{'─'*60}")

        avg_loss = 0.0
        avg_iou = 0.0

        for epoch in range(num_epochs):
            total_loss, total_iou, num_batches = 0.0, 0.0, 0

            for images, masks, filenames in tqdm(task_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                images = images.to(self.device)
                targets = masks.squeeze(1).long().to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss_ce = self.criterion(logits, targets)
                total_loss_value = loss_ce

                if len(self.buffer) > 0:
                    batch_size = min(images.shape[0] // 2, len(self.buffer))
                    buffer_samples = random.sample(self.buffer, batch_size)

                    replay_images = torch.stack([s['image'] for s in buffer_samples]).to(self.device)
                    replay_masks = torch.stack([s['mask'] for s in buffer_samples]).to(self.device)
                    replay_logits = torch.stack([s['logits'] for s in buffer_samples]).to(self.device)

                    current_replay_logits = self.model(replay_images)

                    replay_targets = replay_masks.squeeze(1).long()
                    loss_replay_ce = self.criterion(current_replay_logits, replay_targets)

                    loss_distill = nn.functional.mse_loss(current_replay_logits, replay_logits)

                    total_loss_value = loss_ce + self.beta * loss_replay_ce + self.alpha * loss_distill

                total_loss_value.backward()
                self.optimizer.step()

                predictions = torch.argmax(logits, dim=1)
                _, iou = compute_iou(predictions, targets)

                total_loss += total_loss_value.item()
                total_iou += iou
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_iou = total_iou / num_batches
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, IoU={avg_iou:.4f}")

        return {'loss': avg_loss, 'iou': avg_iou}

    def after_task(self, task_loader, task_id):
        self.model.eval()
        samples_added = 0

        with torch.no_grad():
            for images, masks, filenames in task_loader:
                if samples_added >= self.samples_per_task:
                    break

                images_gpu = images.to(self.device)
                logits = self.model(images_gpu)

                # Store (image, mask, logits)
                for i in range(images.shape[0]):
                    if len(self.buffer) < self.buffer_size:
                        self.buffer.append({
                            'image': images[i].cpu(),
                            'mask': masks[i].cpu(),
                            'logits': logits[i].cpu(),
                            'task_id': task_id
                        })
                    else:
                        # Reservoir sampling
                        idx = random.randint(0, len(self.buffer) - 1)
                        self.buffer[idx] = {
                            'image': images[i].cpu(),
                            'mask': masks[i].cpu(),
                            'logits': logits[i].cpu(),
                            'task_id': task_id
                        }

                    samples_added += 1
                    if samples_added >= self.samples_per_task:
                        break

        self.model.train()
