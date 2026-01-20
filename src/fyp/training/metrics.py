import torch
import numpy as np

def compute_iou(predictions, targets, num_classes=2):
    """Compute mean IoU across classes"""
    ious = []
    for cls in range(num_classes):
        pred_mask = (predictions == cls)
        true_mask = (targets == cls)
        intersection = (pred_mask & true_mask).sum().float()
        union = (pred_mask | true_mask).sum().float()
        iou = (intersection / union).item() if union > 0 else float('nan')
        ious.append(iou)
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return ious, np.mean(valid_ious) if valid_ious else 0.0


def compute_pixel_accuracy(predictions, targets):
    """Compute pixel-wise accuracy"""
    correct = (predictions == targets).sum().float()
    return (correct / targets.numel()).item()
