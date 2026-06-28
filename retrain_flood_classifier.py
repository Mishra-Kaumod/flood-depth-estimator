#!/usr/bin/env python3
"""
Retrain flood classifier on proper dry/wet dataset with better architecture.
"""
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import os
from pathlib import Path
import json

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

TRAIN_DRY_DIR = "flood_dataset/train/dry"
TRAIN_FLOOD_DIR = "flood_dataset/train/flood"
VAL_DRY_DIR = "flood_dataset/val/dry"
VAL_FLOOD_DIR = "flood_dataset/val/flood"
MODEL_SAVE_PATH = "lightweight_flood_classifier_improved.pt"

def create_dataset_loaders(batch_size=32):
    """Create train and validation dataloaders."""
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Collect image paths
    train_images = []
    train_labels = []
    
    # Dry images (label 0)
    for img_path in Path(TRAIN_DRY_DIR).glob("*.jpg"):
        train_images.append(str(img_path))
        train_labels.append(0)
    
    # Flood images (label 1)
    for img_path in Path(TRAIN_FLOOD_DIR).glob("*.jpg"):
        train_images.append(str(img_path))
        train_labels.append(1)
    
    print(f"✅ Train dataset: {len(train_images)} images ({train_labels.count(0)} dry, {train_labels.count(1)} flood)")
    
    # Validation images
    val_images = []
    val_labels = []
    
    for img_path in Path(VAL_DRY_DIR).glob("*.jpg"):
        val_images.append(str(img_path))
        val_labels.append(0)
    
    for img_path in Path(VAL_FLOOD_DIR).glob("*.jpg"):
        val_images.append(str(img_path))
        val_labels.append(1)
    
    print(f"✅ Val dataset: {len(val_images)} images ({val_labels.count(0)} dry, {val_labels.count(1)} flood)")
    
    # Create PyTorch datasets
    class SimpleImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, labels, transform):
            self.image_paths = image_paths
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            from PIL import Image
            img = Image.open(self.image_paths[idx]).convert('RGB')
            img = self.transform(img)
            label = self.labels[idx]
            return img, label
    
    train_dataset = SimpleImageDataset(train_images, train_labels, transform)
    val_dataset = SimpleImageDataset(val_images, val_labels, val_transform)
    
    class_counts = {0: train_labels.count(0), 1: train_labels.count(1)}
    sample_weights = [1.0 / class_counts[label] for label in train_labels]
    train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, class_counts

def build_model():
    """Build MobileNetV3 Small for flood classification."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Replace classifier for binary classification
    model.classifier = nn.Sequential(
        nn.Linear(576, 256),
        nn.BatchNorm1d(256),
        nn.Hardswish(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.Hardswish(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(128, 2),  # 2 classes: dry (0), flood (1)
    )
    
    return model

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    tp = fp = fn = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return avg_loss, accuracy, f1

def main():
    print("\n" + "="*70)
    print("RETRAINING FLOOD CLASSIFIER")
    print("="*70 + "\n")
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}\n")
    
    # Load data
    print("📦 Loading datasets...")
    train_loader, val_loader, class_counts = create_dataset_loaders(batch_size=16)
    
    # Build model
    print("\n🏗️  Building model...")
    model = build_model()
    model = model.to(device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=3)
    total = class_counts[0] + class_counts[1]
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([
            total / (2.0 * max(class_counts[0], 1)),
            total / (2.0 * max(class_counts[1], 1)),
        ]).to(device)
    )
    
    # Training loop
    print("\n🚀 Starting training...\n")
    best_val_f1 = 0
    patience = 0
    
    for epoch in range(30):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:2d}/30 | "
              f"Train: loss={train_loss:.4f}, acc={train_acc:.1f}% | "
              f"Val: loss={val_loss:.4f}, acc={val_acc:.1f}%, f1={val_f1:.3f}")
        
        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"           ✅ Best model saved (f1={val_f1:.3f})")
        else:
            patience += 1
            if patience >= 5:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "="*70)
    print(f"✅ TRAINING COMPLETE - Model saved to {MODEL_SAVE_PATH}")
    print(f"📊 Best validation F1: {best_val_f1:.3f}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
