import copy
from tqdm import tqdm
import torch
import torch.nn.functional as F
from .base import BaseCLStrategy
from ...training.metrics import compute_iou
from ...config import NUM_EPOCHS, DEVICE, SLCA_CA_EPOCHS, SLCA_LOGIT_NORM, IMAGE_SIZE, PATCH_SIZE

class SlowLearnerClassifierAlignment(BaseCLStrategy):
    """
    SLCA: Slow Learner with Classifier Alignment — adapted from Mammoth for segmentation.
    Zhang et al., 2023.

    Since the backbone is frozen, we only perform the Classifier Alignment (CA) phase
    after each task. Synthetic prototypes are generated from the frozen backbone
    features and used to re-align the decoder head in a balanced multi-class batch
    to counter recency bias.
    """

    def __init__(self, model, device=DEVICE, lr=1e-3,
                 ca_epochs=SLCA_CA_EPOCHS,
                 logit_norm=SLCA_LOGIT_NORM):
        super().__init__(model, device, lr)
        self.name = "SLCA"
        self.ca_epochs = ca_epochs
        self.logit_norm = logit_norm

        self.class_means = {}
        self.seen_tasks = []

    def _extract_backbone_features(self, images):
        """Extract spatial features from frozen backbone."""
        with torch.no_grad():
            features = self.model.backbone(images)
        return features

    def _compute_class_prototypes(self, task_loader, task_classes):
        """
        Compute mean feature prototype per class from the frozen backbone.
        Operates pixel-wise on spatial feature maps.
        """
        self.model.eval()
        feature_sums = {c: None for c in task_classes}
        feature_counts = {c: 0 for c in task_classes}

        with torch.no_grad():
            for images, masks, _ in tqdm(task_loader, desc="Computing prototypes", leave=False):
                images = images.to(self.device)
                masks = masks.squeeze(1).long().to(self.device)

                features = self._extract_backbone_features(images)
                B, C, H_f, W_f = features.shape

                masks_resized = F.interpolate(
                    masks.float().unsqueeze(1), size=(H_f, W_f),
                    mode='nearest'
                ).squeeze(1).long()

                for c in task_classes:
                    pixel_mask = (masks_resized == c)
                    if pixel_mask.sum() == 0:
                        continue
                    
                    pixel_feats = features.permute(0, 2, 3, 1)[pixel_mask]
                    if feature_sums[c] is None:
                        feature_sums[c] = pixel_feats.sum(dim=0)
                    else:
                        feature_sums[c] += pixel_feats.sum(dim=0)
                    feature_counts[c] += pixel_mask.sum().item()

        prototypes = {}
        for c in task_classes:
            if feature_counts[c] > 0:
                prototypes[c] = (feature_sums[c] / feature_counts[c]).cpu()

        self.model.train()
        return prototypes

    def _classifier_alignment(self, task_id):
        """
        CA phase: re-balance the segmentation head using stored class prototypes.
        Updates the decoder head using a single multi-class batch containing all
        seen class prototypes expanded to correct spatial dimensions.
        """
        if len(self.class_means) < 2:
            return

        print(f"  Running Classifier Alignment over {len(self.class_means)} classes...")

        ca_optimizer = torch.optim.Adam(
            [p for p in self.model.head.parameters() if p.requires_grad],
            lr=self.optimizer.param_groups[0]['lr'] * 0.1
        )

        class_ids = sorted(self.class_means.keys())
        
        # Stack prototypes and expand to match target feature map shape (N_seen, C, H_f, W_f)
        protos_stacked = torch.stack([self.class_means[c] for c in class_ids]).to(self.device)
        H_f = W_f = IMAGE_SIZE // PATCH_SIZE
        proto_spatial = protos_stacked.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H_f, W_f)

        for ca_epoch in range(self.ca_epochs):
            ca_optimizer.zero_grad()
            
            # Forward directly through the decoder head
            logits = self.model.head(proto_spatial)

            if self.logit_norm > 0:
                norm = torch.norm(logits, p=2, dim=1, keepdim=True).clamp(min=1e-8)
                logits = logits / norm * self.logit_norm

            H_out, W_out = logits.shape[2], logits.shape[3]
            targets = torch.tensor(class_ids, dtype=torch.long, device=self.device)
            targets = targets.view(-1, 1, 1).expand(-1, H_out, W_out)

            loss = self.criterion(logits, targets)
            loss.backward()
            ca_optimizer.step()

            print(f"    CA Epoch {ca_epoch+1}/{self.ca_epochs}: Loss={loss.item():.4f}")

    def train_task(self, task_loader, task_id, num_epochs=NUM_EPOCHS):
        self.model.train()

        print(f"Training Task {task_id+1} - {self.name}")
        print(f"ca_epochs={self.ca_epochs}, logit_norm={self.logit_norm}")
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
                loss = self.criterion(logits, targets)
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
        """
        Extract class prototypes for background (0) and new classes,
        then run Decoder/Classifier Alignment.
        """
        task_classes = [0, task_id + 1]

        print(f"\nAfter Task {task_id+1}: computing prototypes for classes {task_classes}")
        new_prototypes = self._compute_class_prototypes(task_loader, task_classes)
        self.class_means.update(new_prototypes)
        self.seen_tasks.append(task_id)

        if task_id > 0:
            self._classifier_alignment(task_id)
