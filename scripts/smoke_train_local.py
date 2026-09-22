"""
Local smoke training script: small deterministic run using training_data/labels.csv
Saves a candidate model to models/candidate/ for inspection.
Run: py -3 scripts\smoke_train_local.py --samples 120 --epochs 1 --batch-size 8
"""
from pathlib import Path
import argparse
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=120, help="Number of samples to use (random sample from labels.csv)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out-dir", default="models\\candidate")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class LocalFloodDataset(Dataset):
    def __init__(self, df, images_dir, transform):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        p = self.images_dir / row["filename"]
        img = Image.open(p).convert("RGB")
        x = self.transform(img)
        y = float(row["depth_cm"]) / 100.0
        return x, torch.tensor(y, dtype=torch.float32)


def build_model():
    model = models.efficientnet_b0(weights=None)
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_feats, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    return model


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "training_data"
    images_dir = data_dir / "images"
    labels_csv = data_dir / "labels.csv"

    print("Repo root:", repo_root)
    print("Images dir:", images_dir)
    print("Labels CSV:", labels_csv)

    if not labels_csv.exists():
        raise RuntimeError(f"labels.csv not found at {labels_csv}")
    if not images_dir.exists():
        raise RuntimeError(f"images dir not found at {images_dir}")

    df = pd.read_csv(labels_csv)
    df = df.copy()
    df["depth_cm"] = df["depth_cm"].astype(float)

    # filter to existant files
    exists_mask = df["filename"].apply(lambda fn: (images_dir / fn).exists())
    missing = (~exists_mask).sum()
    print(f"Total rows in labels.csv: {len(df)}; missing files referenced: {missing}")
    df = df[exists_mask]
    if df.empty:
        raise RuntimeError("No label rows with existing image files.")

    # bins for distribution
    bins = [-1, 5, 20, 50, 80, 1e9]
    labels = ["0-5", "5-20", "20-50", "50-80", "80+"]
    df["bin"] = pd.cut(df["depth_cm"], bins=bins, labels=labels)
    counts = df["bin"].value_counts().reindex(labels).fillna(0).astype(int)
    print("Distribution by bin:\n", counts)

    n_samples = min(args.samples, len(df))
    df_sample = df.sample(n=n_samples, random_state=args.seed).reset_index(drop=True)
    print(f"Using sample of {n_samples} rows for smoke run")

    # simple splits 80/10/10
    if n_samples >= 10:
        n_train = int(0.8 * n_samples)
        n_val = int(0.1 * n_samples)
    else:
        n_train = max(1, n_samples - 2)
        n_val = 1
    n_test = n_samples - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > 1:
            n_train -= 1

    train_df = df_sample.iloc[:n_train].reset_index(drop=True)
    val_df = df_sample.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df_sample.iloc[n_train + n_val :].reset_index(drop=True)

    print("Split sizes -> train, val, test:", len(train_df), len(val_df), len(test_df))
    print("Train bin counts:\n", train_df["bin"].value_counts().reindex(labels).fillna(0).astype(int))

    # rebalance training by oversampling to per-bin target
    nbins = len(labels)
    target_per_bin = max(1, int(len(train_df) / nbins))
    dfs = []
    for lbl in labels:
        grp = train_df[train_df["bin"] == lbl]
        if len(grp) == 0:
            continue
        if len(grp) < target_per_bin:
            extra = grp.sample(target_per_bin - len(grp), replace=True, random_state=args.seed)
            grp = pd.concat([grp, extra])
        else:
            grp = grp.sample(target_per_bin, replace=False, random_state=args.seed)
        dfs.append(grp)
    if dfs:
        train_bal = pd.concat(dfs).sample(frac=1, random_state=args.seed).reset_index(drop=True)
        print("Balanced train bin counts:\n", train_bal["bin"].value_counts().reindex(labels).fillna(0).astype(int))
    else:
        train_bal = train_df

    # transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = LocalFloodDataset(train_bal, images_dir, transform)
    val_ds = LocalFloodDataset(val_df, images_dir, transform)
    test_ds = LocalFloodDataset(test_df, images_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = build_model().to(device)
    # freeze features
    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.05)

    # train loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        elapsed = time.time() - t0
        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch+1}/{args.epochs} - train loss: {avg_loss:.5f} - time {elapsed:.1f}s")

        # val MAE
        model.eval()
        preds = []
        gts = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb).cpu().numpy().reshape(-1)
                preds.append(out)
                gts.append(yb.numpy())
        preds = np.concatenate(preds) if preds else np.array([])
        gts = np.concatenate(gts) if gts else np.array([])
        if len(preds):
            mae = mean_absolute_error(gts * 100.0, preds * 100.0)
            print(f"Val MAE: {mae:.2f} cm")

    # final test eval
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy().reshape(-1)
            preds.append(out)
            gts.append(yb.numpy())
    preds = np.concatenate(preds) if preds else np.array([])
    gts = np.concatenate(gts) if gts else np.array([])
    if len(preds):
        mae = mean_absolute_error(gts * 100.0, preds * 100.0)
        rmse = mean_squared_error(gts * 100.0, preds * 100.0)
        rmse = float(np.sqrt(rmse))
        r2 = r2_score(gts * 100.0, preds * 100.0)
        print(f"Test MAE: {mae:.2f} cm | RMSE: {rmse:.2f} cm | R2: {r2:.3f}")

    # save model
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"smoke_trained_{ts}.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs}, str(out_path))
    print("Saved model at:", out_path)


if __name__ == '__main__':
    main()
