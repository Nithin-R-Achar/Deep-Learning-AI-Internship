import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import MNISTNet

def train_model(epochs=5, batch_size=128, lr=0.001, model_save_path="mnist_cnn.pth", meta_save_path="model_meta.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data augmentation for training to make model resilient to varied hand-drawn digits
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("Loading MNIST dataset...")
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = MNISTNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": []
    }

    best_acc = 0.0
    start_time = time.time()

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 150 == 0 or (batch_idx + 1) == total_batches:
                print(f"Epoch [{epoch}/{epochs}] | Step [{batch_idx + 1}/{total_batches}] | Loss: {loss.item():.4f}")

        scheduler.step()
        epoch_train_loss = running_loss / total_batches

        # Evaluation
        model.eval()
        test_loss = 0.0
        correct = 0
        total_test = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total_test += data.size(0)

        epoch_test_loss = test_loss / total_test
        epoch_test_acc = (correct / total_test) * 100.0

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["test_loss"].append(round(epoch_test_loss, 4))
        history["test_acc"].append(round(epoch_test_acc, 2))

        print(f"--> Epoch {epoch} Results: Test Loss: {epoch_test_loss:.4f} | Test Accuracy: {epoch_test_acc:.2f}%")

        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model checkpoint to {model_save_path} ({best_acc:.2f}%)")

    total_duration = time.time() - start_time
    print(f"\nTraining completed in {total_duration:.1f} seconds. Best Accuracy: {best_acc:.2f}%")

    metadata = {
        "best_accuracy": best_acc,
        "epochs": epochs,
        "batch_size": batch_size,
        "history": history,
        "training_time_seconds": round(total_duration, 2),
        "model_architecture": "MNISTNet (2 Conv Blocks + BatchNorm + Dropout + Dense)"
    }
    with open(meta_save_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {meta_save_path}")
    return model, metadata

if __name__ == "__main__":
    train_model(epochs=5, batch_size=128, lr=0.001)
