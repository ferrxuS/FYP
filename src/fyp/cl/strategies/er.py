from tqdm import tqdm
import torch
from .base import BaseCLStrategy
from ..buffer import ReplayBuffer
from ...training.metrics import compute_iou
from ...config import NUM_EPOCHS, MEMORY_SIZE, DEVICE

class ExperienceReplay(BaseCLStrategy):
    """Experience Replay: Store samples from old tasks and replay during training"""

    def __init__(self, model, device=DEVICE, lr=1e-3,
                 buffer_size=MEMORY_SIZE, samples_per_task=100):
        super().__init__(model, device, lr)
        self.name = "Experience Replay (ER)"
        self.buffer = ReplayBuffer(max_size=buffer_size)
        self.samples_per_task = samples_per_task

    def train_task(self, task_loader, task_id, num_epochs=NUM_EPOCHS):
        """Train on current task + replay samples"""
        self.model.train()

        print(f"Training Task {task_id+1} - {self.name}")
        print(f"Buffer: {len(self.buffer)}/{self.buffer.max_size} samples")

        avg_loss = 0.0
        avg_iou = 0.0

        for epoch in range(num_epochs):
            total_loss, total_iou, num_batches = 0.0, 0.0, 0

            for images, masks, filenames in tqdm(task_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                images = images.to(self.device)
                targets = masks.squeeze(1).long().to(self.device)

                # Get replay samples if buffer not empty
                if len(self.buffer) > 0:
                    replay_images, replay_masks = self.buffer.sample(
                        batch_size=images.shape[0] // 2,  # Half batch from buffer
                        device=self.device
                    )

                    if replay_images is not None:
                        images = torch.cat([images, replay_images], dim=0)
                        replay_targets = replay_masks.squeeze(1).long()
                        targets = torch.cat([targets, replay_targets], dim=0)

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
        samples_added = 0
        for images, masks, filenames in task_loader:
            if samples_added >= self.samples_per_task:
                break

            self.buffer.add_samples(images, masks, filenames, task_id)
            samples_added += images.shape[0]
