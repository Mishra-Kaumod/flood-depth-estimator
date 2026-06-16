import os
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.model_selection import train_test_split

# =====================================================
# CONFIG
# =====================================================

CSV_FILE = "/workspaces/flood-depth-estimator/data/benchmark_labels_clean.csv"
IMAGE_DIR = "/workspaces/flood-depth-estimator/data/floodnet/train/images"

MODEL_PATH = "/workspaces/flood-depth-estimator/models/severity_efficientnet.pth"

BATCH_SIZE = 4
EPOCHS = 20

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(CSV_FILE)

df["image_name"] = df["image_name"].astype(str).str.strip()

available_files = set(os.listdir(IMAGE_DIR))

df = df[df["image_name"].isin(available_files)]

print("Matched Images:", len(df))

print("\nSeverity Distribution")
print(df["severity"].value_counts().sort_index())

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["severity"],
    random_state=42
)

# =====================================================
# TRANSFORMS
# =====================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =====================================================
# DATASET
# =====================================================

class FloodDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = os.path.join(
            IMAGE_DIR,
            row["image_name"]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.transform(image)

        label = int(row["severity"])

        return image, label


train_dataset = FloodDataset(
    train_df,
    train_transform
)

val_dataset = FloodDataset(
    val_df,
    val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE
)

# =====================================================
# MODEL
# =====================================================

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(
    weights=weights
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    5
)

model = model.to(device)

# =====================================================
# CLASS WEIGHTS
# =====================================================

counts = (
    df["severity"]
    .value_counts()
    .sort_index()
)

class_weights = torch.tensor(
    [
        1.0 / counts[0],
        1.0 / counts[1],
        1.0 / counts[2],
        1.0 / counts[3],
        1.0 / counts[4]
    ],
    dtype=torch.float
).to(device)

print("\nClass Weights")
print(class_weights)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

# =====================================================
# TRAIN
# =====================================================

best_acc = 0

os.makedirs(
    "/workspaces/flood-depth-estimator/models",
    exist_ok=True
)

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    # VALIDATION

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            preds = outputs.argmax(1)

            correct += (
                preds == labels
            ).sum().item()

            total += labels.size(0)

    val_acc = correct / total

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss={train_loss:.4f} "
        f"ValAcc={val_acc:.4f}"
    )

    if val_acc > best_acc:

        best_acc = val_acc

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"Saved Best Model "
            f"(ValAcc={best_acc:.4f})"
        )

print("\nTraining Complete")
print("Best Accuracy:", best_acc)
print("Saved:", MODEL_PATH)
