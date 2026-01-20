import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..data.dataset import ContainerDefectDataset
from ..models.segmenter import ContainerDefectSegmenter
from .metrics import compute_iou, compute_pixel_accuracy
from ..config import (
    TRAIN_DIR, VALID_DIR, NUM_EPOCHS, BATCH_SIZE, 
    LEARNING_RATE, SUBSET_SIZE, SAVE_DIR, DEVICE
)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss, total_iou, total_acc, num_batches = 0.0, 0.0, 0.0, 0

    for images, masks, _ in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        targets = masks.squeeze(1).long().to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute metrics
        predictions = torch.argmax(logits, dim=1)
        _, iou = compute_iou(predictions, targets)
        acc = compute_pixel_accuracy(predictions, targets)

        # Accumulate
        total_loss += loss.item()
        total_iou += iou
        total_acc += acc
        num_batches += 1

    return total_loss / num_batches, total_iou / num_batches, total_acc / num_batches

@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss, total_iou, total_acc, num_batches = 0.0, 0.0, 0.0, 0

    for images, masks, _ in tqdm(dataloader, desc="Validation", leave=False):
        images = images.to(device)
        targets = masks.squeeze(1).long().to(device)

        # Forward pass
        logits = model(images)

        # Compute loss
        loss = criterion(logits, targets)

        # Compute metrics
        predictions = torch.argmax(logits, dim=1)
        _, iou = compute_iou(predictions, targets)
        acc = compute_pixel_accuracy(predictions, targets)

        # Accumulate
        total_loss += loss.item()
        total_iou += iou
        total_acc += acc
        num_batches += 1

    return total_loss / num_batches, total_iou / num_batches, total_acc / num_batches

def train_baseline_model(
    train_dir=TRAIN_DIR, valid_dir=VALID_DIR,
    num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
    subset_size=SUBSET_SIZE, save_path=None
):
    """Train the baseline model (*all data at once, no CL*)"""
    if save_path is None:
        save_path = SAVE_DIR / "baseline_model.pth"
    
    # Datasets
    train_dataset = ContainerDefectDataset(train_dir, subset_size=subset_size)

    val_dataset = ContainerDefectDataset(
        valid_dir,
        subset_size=subset_size // 2 if subset_size else None
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Model
    model = ContainerDefectSegmenter().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    print(f"\nBASELINE TRAINING\n")
    print(f"Device: {DEVICE}")
    print(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    best_val_iou = 0.0
    for epoch in range(num_epochs):
        # Train
        train_loss, train_iou, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        # Validate
        val_loss, val_iou, val_acc = validate(
            model, val_loader, criterion, DEVICE
        )
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train:  Loss={train_loss:.4f}, IoU={train_iou:.4f}, Acc={train_acc:.4f}")
        print(f"  Val:    Loss={val_loss:.4f}, IoU={val_iou:.4f}, Acc={val_acc:.4f}")

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (IoU: {val_iou:.4f})")

    print("\n" + "=" * 60)
    print(f"Training complete | Best IoU: {best_val_iou:.4f}")
    print(f"Model saved: {save_path}")
    return model
