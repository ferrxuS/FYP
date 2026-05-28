from tqdm import tqdm
import torch
import torch.nn as nn
from .base import BaseCLStrategy
from ...training.metrics import compute_iou
from ...config import NUM_EPOCHS, DEVICE, SI_C, SI_XI

class SynapticIntelligence(BaseCLStrategy):
    """
    SI: Synaptic Intelligence — faithful port of Mammoth's implementation.
    Zenke et al., 2017 — "Continual Learning Through Synaptic Intelligence"

    Regularization-based CL: tracks per-parameter importance (big_omega)
    accumulated over training trajectory, penalises changes to important
    parameters when learning new tasks. No replay buffer required.
    """

    def __init__(self, model, device=DEVICE, lr=1e-3,
                 c=SI_C, xi=SI_XI):
        super().__init__(model, device, lr)
        self.name = "Synaptic Intelligence (SI)"
        self.c = c      # surrogate loss weight — scales penalty strength
        self.xi = xi    # damping term — prevents division by zero in omega update

        # Get list of parameters requiring gradients
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Parameter-wise buffers (OOM-proof, no massive flat tensors created in training loop)
        self.checkpoint = {p: p.data.clone() for p in self.trainable_params}
        self.big_omega = {}     # accumulated importance weights (Ω)
        self.small_omega = {p: torch.zeros_like(p.data) for p in self.trainable_params}
        self.has_big_omega = False

    def train_task(self, task_loader, task_id, num_epochs=NUM_EPOCHS):
        """Train task while tracking parameter contributions and penalizing changes to important weights"""
        self.model.train()

        print(f"Training Task {task_id+1} - {self.name}")
        print(f"c={self.c}, xi={self.xi}")
        print(f"{'─'*60}")

        avg_loss = 0.0
        avg_iou = 0.0

        for epoch in range(num_epochs):
            total_loss, total_iou, num_batches = 0.0, 0.0, 0

            for images, masks, filenames in tqdm(task_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                images = images.to(self.device)
                targets = masks.squeeze(1).long().to(self.device)

                # Snapshot parameter data before step (needed for small_omega update)
                pre_params = {p: p.data.clone() for p in self.trainable_params}

                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, targets)
                loss.backward()

                # Capture raw task gradients immediately after backward, before penalty injection or optimizer step
                raw_grads = {p: p.grad.data.clone() for p in self.trainable_params if p.grad is not None}

                # Inject penalty gradients into existing grads (Tasks 2+)
                if self.has_big_omega:
                    for p in self.trainable_params:
                        if p.grad is not None:
                            # penalty grad = c * 2 * big_omega * (theta - theta*)
                            penalty_g = self.c * 2 * self.big_omega[p] * (p.data - self.checkpoint[p])
                            p.grad.data.add_(penalty_g)

                # Gradient value clipping (Mammoth uses clip_grad_value_=1.0)
                torch.nn.utils.clip_grad_value_(self.trainable_params, 1.0)
                self.optimizer.step()

                # Accumulate small_omega: ω += g * Δθ (gradient dotted with parameter displacement)
                # Done parameter-wise to avoid flattening overhead, using raw captured gradients
                for p in self.trainable_params:
                    if p in raw_grads:
                        delta_theta = pre_params[p] - p.data
                        self.small_omega[p] += raw_grads[p] * delta_theta

                # Metrics
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
        """
        End-of-task Ω update (Mammoth's end_task):
        Ω += ω / ((θ - θ*)² + ξ)
        Then reset θ* and ω for the next task.
        """
        if not self.has_big_omega:
            self.big_omega = {p: torch.zeros_like(p.data) for p in self.trainable_params}
            self.has_big_omega = True

        for p in self.trainable_params:
            # Ω += ω / ((θ - θ*)² + ξ)
            displacement = (p.data - self.checkpoint[p]) ** 2
            self.big_omega[p] += self.small_omega[p] / (displacement + self.xi)

            # Reset checkpoints and small omegas
            self.checkpoint[p].copy_(p.data)
            self.small_omega[p].zero_()
