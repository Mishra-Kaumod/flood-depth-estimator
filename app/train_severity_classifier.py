from pathlib import Path
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Load labels
df = pd.read_csv("severity_labels.csv")

# Build image lookup
image_paths = {}
for split in ["train", "val", "test"]:
    folder = Path(f"datasets/flooddet/extracted/FloodDET/{split}")
    for img in folder.glob("*.jpg"):
        image_paths[img.name] = str(img)

df["path"] = df["image"].map(image_paths)
df = df.dropna()

class FloodDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe
        self.tf = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["path"]).convert("RGB")
        img = self.tf(img)

        label = int(row["severity"])

        return img, label

dataset = FloodDataset(df)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True
)

model = models.resnet18(weights="DEFAULT")

model.fc = nn.Linear(
    model.fc.in_features,
    5
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4
)

print("Training started...")

for epoch in range(5):

    model.train()

    running_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}: "
        f"Loss={running_loss/len(loader):.4f}"
    )

torch.save(
    model.state_dict(),
    "severity_model.pth"
)

print("Model saved as severity_model.pth")