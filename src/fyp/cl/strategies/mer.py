from tqdm import tqdm
import torch
from .base import BaseCLStrategy
from ..buffer import ReplayBuffer
from ...training.metrics import compute_iou
from ...config import NUM_EPOCHS, MEMORY_SIZE, DEVICE, MER_BETA, MER_GAMMA, MER_BATCH_NUM

class MetaExperienceReplay(BaseCLStrategy):
    """
    MER: Meta-Experience Replay — faithful port of Mammoth's Alg 6.
    Two-level Reptile: within-batch (beta) + across-batch (gamma).
    Riemer et al., 2019.
    """

    def __init__(self, model, device=DEVICE, lr=1e-3,
                 buffer_size=MEMORY_SIZE, samples_per_task=100,
                 beta=MER_BETA,
                 gamma=MER_GAMMA,
                 batch_num=MER_BATCH_NUM):
        super().__init__(model, device, lr)
        self.name = "Meta Experience Replay (MER)"
        self.buffer = ReplayBuffer(max_size=buffer_size)
        self.samples_per_task = samples_per_task
        self.beta = beta
        self.gamma = gamma
        self.batch_num = batch_num

    def train_task(self, task_loader, task_id, num_epochs=NUM_EPOCHS):
        """Train on current task + replay using Mammoth's two-level Reptile updates"""
        self.model.train()

        print(f"Training Task {task_id+1} - {self.name}")
        print(f"Buffer: {len(self.buffer)}/{self.buffer.max_size} samples")
        print(f"beta={self.beta}, gamma={self.gamma}, batch_num={self.batch_num}")

        avg_loss = 0.0
        avg_iou = 0.0

        # Filter parameters that require gradients
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        for epoch in range(num_epochs):
            total_loss, total_iou, num_batches = 0.0, 0.0, 0

            for images, masks, filenames in tqdm(task_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                images = images.to(self.device)
                targets = masks.squeeze(1).long().to(self.device)

                # --- 1. Level 2 across-batch anchor (gamma) ---
                # Done parameter-wise in-place to avoid massive contiguous flat tensor allocation in GPU
                theta_A0 = [p.data.clone() for p in trainable_params]

                for _ in range(self.batch_num):
                    # --- 2. Level 1 within-batch anchor (beta) ---
                    theta_Wi0 = [p.data.clone() for p in trainable_params]

                    # Construct combined batch: concat buffer + current task data
                    if len(self.buffer) > 0:
                        replay_batch_size = max(1, images.shape[0] // 2)
                        buf_images, buf_masks = self.buffer.sample(
                            batch_size=replay_batch_size,
                            device=self.device
                        )
                        if buf_images is not None:
                            batch_images = torch.cat([buf_images, images], dim=0)
                            buf_targets = buf_masks.squeeze(1).long().to(self.device)
                            batch_targets = torch.cat([buf_targets, targets], dim=0)
                        else:
                            batch_images, batch_targets = images, targets
                    else:
                        batch_images, batch_targets = images, targets

                    # Within-batch step
                    self.optimizer.zero_grad()
                    logits = self.model(batch_images)
                    loss_val = self.criterion(logits, batch_targets)
                    loss_val.backward()
                    # Clip gradients to ensure stable weight adjustments in ViT/DINOv3 backbones
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    # Within-batch Reptile meta-update (beta)
                    for p, w_i in zip(trainable_params, theta_Wi0):
                        p.data.copy_(w_i + self.beta * (p.data - w_i))

                # --- 3. Level 2 across-batch Reptile meta-update (gamma) ---
                for p, w_a in zip(trainable_params, theta_A0):
                    p.data.copy_(w_a + self.gamma * (p.data - w_a))

                # 4. Evaluate metrics on the current batch after full meta-updates
                with torch.no_grad():
                    logits = self.model(images)
                    loss_eval = self.criterion(logits, targets)
                    predictions = torch.argmax(logits, dim=1)
                    _, iou = compute_iou(predictions, targets)

                total_loss += loss_eval.item()
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
