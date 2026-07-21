"""
Simple one-file Colab retraining script (upload + Kaggle + Gemini + GPU).

Run in Colab:
  !pip install -q torch torchvision pillow tqdm kaggle google-genai
  !python retrain_flood_classifier.py --repo-dir /content/flood-depth-estimator --version v3
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import random
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from tqdm import tqdm
except Exception:
    def tqdm(items, **_kwargs):  # type: ignore
        return items

try:
    from google.colab import files  # type: ignore

    IN_COLAB = True
except Exception:
    IN_COLAB = False

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_cmd(command: Sequence[str], cwd: Optional[Path] = None) -> None:
    result = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def find_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if is_image(p)])


def build_model() -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    return model


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        candidate = checkpoint.get("model_state_dict", checkpoint)
        if isinstance(candidate, dict):
            return candidate
    if isinstance(checkpoint, dict):
        return checkpoint
    raise RuntimeError("Unsupported checkpoint format.")


def load_base_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> Dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = _extract_state_dict(checkpoint)
    model_state = model.state_dict()

    matched = {k: v for k, v in state.items() if k in model_state and model_state[k].shape == v.shape}

    # Optional remap for older naming conventions.
    if not matched:
        remapped: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            mapped: Optional[str] = None
            if key.startswith("backbone."):
                mapped = "features." + key[len("backbone.") :]
            elif key.startswith("head.0."):
                mapped = key.replace("head.0.", "classifier.1.")
            elif key.startswith("head.4."):
                mapped = key.replace("head.4.", "classifier.4.")
            elif key.startswith("head.8."):
                mapped = key.replace("head.8.", "classifier.6.")
            if mapped and mapped in model_state and model_state[mapped].shape == value.shape:
                remapped[mapped] = value
        matched = remapped

    if not matched:
        raise RuntimeError(
            "Failed to load checkpoint weights into current model. "
            "Please upload latest model file at models/best_flood_model_water_aware.pth."
        )

    model_state.update(matched)
    model.load_state_dict(model_state)
    print(f"Loaded {len(matched)} / {len(model_state)} tensors from base checkpoint.")
    return checkpoint if isinstance(checkpoint, dict) else {}


def write_kaggle_json(username: str, key: str) -> None:
    target = Path.home() / ".kaggle" / "kaggle.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(target, 0o600)


def setup_kaggle_credentials(kaggle_username: str, kaggle_key: str) -> None:
    existing = Path.home() / ".kaggle" / "kaggle.json"
    if existing.exists():
        return

    username = kaggle_username.strip()
    key = kaggle_key.strip()
    if username and key:
        write_kaggle_json(username, key)
        return

    username = input("Kaggle username (press Enter to upload kaggle.json): ").strip()
    if username:
        key = getpass.getpass("Kaggle API key: ").strip()
        if not key:
            raise RuntimeError("Kaggle API key cannot be empty.")
        write_kaggle_json(username, key)
        return

    if not IN_COLAB:
        raise RuntimeError("Provide --kaggle-username and --kaggle-key, or place ~/.kaggle/kaggle.json.")

    uploaded = files.upload()
    if "kaggle.json" not in uploaded:
        raise RuntimeError("kaggle.json upload missing.")
    target = Path.home() / ".kaggle" / "kaggle.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as f:
        f.write(uploaded["kaggle.json"])
    os.chmod(target, 0o600)


def download_kaggle_dataset(slug: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / slug.replace("/", "_")
    target.mkdir(parents=True, exist_ok=True)
    run_cmd(["kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"])
    return target


def _bucket(path: Path) -> str:
    text = str(path.parent).lower()
    if re.search(r"non[-_ ]?flood|no[-_ ]?flood|dry|normal|clear", text):
        return "non_flood"
    if re.search(r"flood|waterlog|inundat|submerg", text):
        return "flood"
    return "other"


def select_balanced(images: List[Path], max_images: int, seed: int) -> List[Path]:
    buckets = {"flood": [], "non_flood": [], "other": []}
    for p in images:
        buckets[_bucket(p)].append(p)
    rng = random.Random(seed)
    for group in buckets.values():
        rng.shuffle(group)

    if max_images <= 0:
        return images

    half = max_images // 2
    selected = buckets["flood"][:half] + buckets["non_flood"][:half]
    remain = max_images - len(selected)
    if remain > 0:
        selected.extend((buckets["other"] + buckets["flood"][half:] + buckets["non_flood"][half:])[:remain])
    return selected


def upload_user_images(dest: Path) -> List[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    if IN_COLAB:
        print("Upload your flood images now.")
        uploaded = files.upload()
        for name, data in uploaded.items():
            with open(dest / name, "wb") as f:
                f.write(data)
        return find_images(dest)
    return find_images(dest)


def create_gemini_callable(api_key: str, model_name: str):
    # Prefer google-genai, fallback to google.generativeai.
    try:
        from google import genai as modern_genai  # type: ignore
        from google.genai import types as modern_types  # type: ignore

        client = modern_genai.Client(api_key=api_key)

        def call(image_path: Path, prompt: str) -> str:
            with open(image_path, "rb") as f:
                payload = f.read()
            mime_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, modern_types.Part.from_bytes(data=payload, mime_type=mime_type)],
            )
            return getattr(response, "text", "") or ""

        return call
    except Exception:
        import google.generativeai as legacy_genai  # type: ignore

        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(model_name)

        def call(image_path: Path, prompt: str) -> str:
            img = Image.open(image_path).convert("RGB")
            response = model.generate_content([prompt, img])
            return getattr(response, "text", "") or ""

        return call


def parse_depth(text: str, default_cm: float) -> float:
    m = re.search(r"(-?\d+(\.\d+)?)", text)
    if not m:
        return default_cm
    return float(np.clip(float(m.group(1)), 0.0, 100.0))


def label_with_gemini(
    images: List[Path],
    output_csv: Path,
    gemini_api_key: str,
    gemini_model: str,
    default_depth_cm: float,
    delay_seconds: float,
) -> None:
    predict_text = create_gemini_callable(gemini_api_key, gemini_model)
    rows: List[Dict[str, str]] = []
    prompt = (
        "Estimate flood water depth in centimeters. "
        "Use visible object-waterline cues (person/car/bike/building). "
        "Return only one number in cm."
    )
    for img_path in tqdm(images, desc="Gemini labeling"):
        try:
            response = predict_text(img_path, prompt)
            depth = parse_depth(response, default_depth_cm)
        except Exception:
            depth = default_depth_cm
        rows.append({"filename": img_path.name, "depth_cm": f"{depth:.2f}"})
        time.sleep(delay_seconds)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "depth_cm"])
        writer.writeheader()
        writer.writerows(rows)


class FloodDepthDataset(Dataset):
    def __init__(self, image_dir: Path, labels_csv: Path):
        self.items: List[Tuple[Path, float]] = []
        with open(labels_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = image_dir / row["filename"]
                if p.exists():
                    self.items.append((p, float(row["depth_cm"])))

        if not self.items:
            raise RuntimeError("No labeled samples found.")

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, depth_cm = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(depth_cm / 100.0, dtype=torch.float32)
        return {"image": x, "target": y}


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Dict:
    # Transfer learning: train classifier only.
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.05)
    best = {"val_loss": float("inf")}

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= max(1, len(train_loader))

        model.eval()
        v_loss = 0.0
        v_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["target"].to(device).unsqueeze(1)
                pred = model(images)
                v_loss += criterion(pred, targets).item()
                v_mae += torch.abs((pred - targets) * 100.0).mean().item()
        v_loss /= max(1, len(val_loader))
        v_mae /= max(1, len(val_loader))
        print(f"Epoch {epoch + 1}/{epochs} | train_loss={t_loss:.5f} | val_loss={v_loss:.5f} | val_mae={v_mae:.2f}cm")

        if v_loss < best["val_loss"]:
            best = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": v_loss,
                "val_mae": v_mae,
            }
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Colab retraining script")
    parser.add_argument("--repo-dir", default=".")
    parser.add_argument("--base-model", default="models/best_flood_model_water_aware.pth")
    parser.add_argument("--kaggle-dataset", default="jannalipka/flood-detection-image-dataset")
    parser.add_argument("--kaggle-username", default="")
    parser.add_argument("--kaggle-key", default="")
    parser.add_argument("--gemini-api-key", default="")
    parser.add_argument("--gemini-model", default="gemini-1.5-flash")
    parser.add_argument("--default-depth-cm", type=float, default=30.0)
    parser.add_argument("--gemini-delay", type=float, default=0.3)
    parser.add_argument("--max-kaggle-images", type=int, default=300)
    parser.add_argument("--max-total-images", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--version", default="v3")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_dir = Path(args.repo_dir).resolve()
    os.chdir(repo_dir)

    print(f"Running in: {repo_dir}")
    print(f"Device: {device}")

    workspace = repo_dir / "colab_simple_run"
    user_dir = workspace / "user_images"
    kaggle_raw = workspace / "kaggle_raw"
    merged_dir = workspace / "training_images"
    workspace.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    user_images = upload_user_images(user_dir)
    print(f"Uploaded user images: {len(user_images)}")

    setup_kaggle_credentials(args.kaggle_username, args.kaggle_key)
    kaggle_root = download_kaggle_dataset(args.kaggle_dataset, kaggle_raw)
    kaggle_images_all = find_images(kaggle_root)
    kaggle_images = select_balanced(kaggle_images_all, args.max_kaggle_images, args.seed)
    print(f"Kaggle images selected: {len(kaggle_images)}")

    all_sources = user_images + kaggle_images
    if args.max_total_images > 0:
        all_sources = all_sources[: args.max_total_images]
    if len(all_sources) < 20:
        raise RuntimeError(f"Need at least 20 images, found {len(all_sources)}")

    copied: List[Path] = []
    for i, src in enumerate(all_sources, start=1):
        tag = "user" if src in user_images else "kaggle"
        dst = merged_dir / f"{tag}_{i:05d}{src.suffix.lower()}"
        shutil.copy2(src, dst)
        copied.append(dst)
    print(f"Total training images: {len(copied)}")

    gemini_key = args.gemini_api_key.strip() or getpass.getpass("Gemini API key: ").strip()
    labels_csv = workspace / "labels.csv"
    label_with_gemini(
        images=copied,
        output_csv=labels_csv,
        gemini_api_key=gemini_key,
        gemini_model=args.gemini_model,
        default_depth_cm=args.default_depth_cm,
        delay_seconds=args.gemini_delay,
    )

    model = build_model().to(device)
    base_ckpt = load_base_checkpoint(model, repo_dir / args.base_model, device)
    if isinstance(base_ckpt, dict):
        print(f"Base model epoch={base_ckpt.get('epoch','n/a')} val_loss={base_ckpt.get('val_loss','n/a')}")

    dataset = FloodDepthDataset(merged_dir, labels_csv)
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.learning_rate,
    )

    model_dir = repo_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_version = model_dir / f"best_flood_model_{args.version}.pth"
    out_default = model_dir / "best_flood_model_water_aware.pth"
    torch.save(best, out_version)
    shutil.copy2(out_version, out_default)

    print(f"Saved: {out_version}")
    print(f"Updated default model: {out_default}")

    if IN_COLAB:
        files.download(str(out_version))

    print("\nRun locally after download:")
    print("python app.py")


if __name__ == "__main__":
    main()
