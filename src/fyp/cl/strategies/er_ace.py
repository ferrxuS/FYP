from tqdm import tqdm
import torch
from .base import BaseCLStrategy
from ..buffer import ReplayBuffer
from ...training.metrics import compute_iou
from ...config import NUM_EPOCHS, MEMORY_SIZE, DEVICE

class ExperienceReplayACE(BaseCLStrategy):
    """
    ER-ACE: Experience Replay with Asymmetric Cross-Entropy.
    Adapted from Mammoth for dense segmentation (B, C, H, W) logits.
    Buzzega et al., 2022.

    Asymmetric loss treatment:
    - Current task samples: CE masked to only seen classes (prevents
      the model from being penalised for not predicting unseen classes)
    - Buffer samples: standard CE (classes were seen during their task)

    Directly addresses background drift in class-incremental segmentation:
    old classes in buffer samples are no longer incorrectly penalised
    as background when new classes are introduced.
    """

    def __init__(self, model, device=DEVICE, lr=1e-3,
                 buffer_size=MEMORY_SIZE, samples_per_task=100):
        super().__init__(model, device, lr)
        self.name = "ER-ACE"
        self.buffer = ReplayBuffer(max_size=buffer_size)
        self.samples_per_task = samples_per_task
        self.seen_classes = set()

    def _ace_loss(self, logits, targets, present_classes):
        """
        Asymmetric CE loss for current task samples.
        Masks out logits for classes not yet seen, preventing the model
        from being penalised for classes it hasn't been trained on yet.
        """
        allowed = self.seen_classes.union(present_classes).union({0})

        mask = torch.zeros(logits.shape[1], device=logits.device)
        for c in allowed:
            if c < logits.shape[1]:
                mask[c] = 1.0

        mask = mask.view(1, -1, 1, 1)
        masked_logits = logits.masked_fill(mask == 0, float('-inf'))

        return self.criterion(masked_logits, targets)

    def train_task(self, task_loader, task_id, num_epochs=NUM_EPOCHS):
        self.model.train()

        print(f"Training Task {task_id+1} - {self.name}")
        print(f"Buffer: {len(self.buffer)}/{self.buffer.max_size} samples")
        print(f"Seen classes so far: {sorted(self.seen_classes)}")
        print(f"{'─'*60}")

        avg_loss = 0.0
        avg_iou = 0.0

        for epoch in range(num_epochs):
            total_loss, total_iou, num_batches = 0.0, 0.0, 0

            for images, masks, filenames in tqdm(task_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                images = images.to(self.device)
                targets = masks.squeeze(1).long().to(self.device)

                present_classes = set(targets.unique().cpu().tolist()) - {0}
                self.seen_classes.update(present_classes)

                self.optimizer.zero_grad()
                logits = self.model(images)

                if task_id > 0:
                    loss = self._ace_loss(logits, targets, present_classes)
                else:
                    loss = self.criterion(logits, targets)

                if len(self.buffer) > 0:
                    replay_batch_size = max(1, images.shape[0] // 2)
                    buf_images, buf_masks = self.buffer.sample(
                        batch_size=replay_batch_size, device=self.device
                    )
                    if buf_images is not None:
                        buf_targets = buf_masks.squeeze(1).long()
                        buf_logits = self.model(buf_images)
                        loss_replay = self.criterion(buf_logits, buf_targets)
                        loss = loss + loss_replay

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    _, iou = compute_iou(preds, targets)

                total_loss += loss.item()
                total_iou += iou
                num_batches += 1

            avg_loss = total_loss / num_batches
            avg_iou = total_iou / num_batches
            print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}, IoU={avg_iou:.4f}")

        return {'loss': avg_loss, 'iou': avg_iou}

    def after_task(self, task_loader, task_id):
        """Fill buffer with reservoir sampling after each task."""
        samples_added = 0
        for images, masks, filenames in task_loader:
            if samples_added >= self.samples_per_task:
                break
            self.buffer.add_samples(images, masks, filenames, task_id)
            samples_added += images.shape[0]
