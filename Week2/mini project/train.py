"""
train.py
Fine-tunes a pre-trained MobileNetV2 on the Plant Leaf Disease dataset.
Saves model weights, training metrics, and class index mapping for deployment.
"""

import os
import sys
import time
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from model import build_model, train_transforms, inference_transforms

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = (correct / total) * 100
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on validation dataset."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    val_loss = running_loss / total if total > 0 else 0.0
    val_acc = (correct / total) * 100 if total > 0 else 0.0
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description="Fine-tune MobileNetV2 on Plant Leaf Diseases")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Directory containing train/ and val/ splits")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model weights")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze lower feature layers")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"[-] Error: Could not find '{train_dir}' or '{val_dir}'. Please run download_dataset.py first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("  PLANT DISEASE DETECTOR - MODEL TRAINING")
    print("=" * 65)
    print(f"Device:           {device}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.lr}")
    print(f"Freeze Backbone:  {args.freeze_backbone}")
    print(f"Model Directory:  {os.path.abspath(args.output_dir)}")
    print("=" * 65)

    # Load datasets
    train_dataset = ImageFolder(train_dir, transform=train_transforms)
    val_dataset = ImageFolder(val_dir, transform=inference_transforms)

    classes = train_dataset.classes
    num_classes = len(classes)
    idx_to_class = {i: c for i, c in enumerate(classes)}

    print(f"\n[+] Loaded {len(train_dataset)} training images across {num_classes} classes.")
    print(f"[+] Loaded {len(val_dataset)} validation images.")

    # Save class indices
    class_indices_path = os.path.join(args.output_dir, "class_indices.json")
    with open(class_indices_path, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"[+] Saved class mapping to {class_indices_path}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Initialize MobileNetV2 with pre-trained ImageNet weights
    print("[+] Building MobileNetV2 model...")
    model = build_model(num_classes=num_classes, pretrained=True, freeze_backbone=args.freeze_backbone)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_model_path = os.path.join(args.output_dir, "plant_disease_model.pth")
    history = []

    print("\n--- Commencing Fine-Tuning ---")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_dur = time.time() - epoch_start

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({epoch_dur:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best checkpoint (Val Acc: {val_acc:.2f}%) to {best_model_path}")

    # If best accuracy was tied or never saved (e.g. all 100%), ensure saved
    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"  TRAINING COMPLETE! Total time: {total_time:.1f}s")
    print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"  Checkpoint saved to:     {best_model_path}")
    print("=" * 65)

    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
