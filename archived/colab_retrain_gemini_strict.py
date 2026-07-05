"""
Strict Colab retraining script:
- Uses current base model: models/best_flood_model_water_aware.pth
- Uses ONLY uploaded new images
- Labels with Gemini (no default/fallback labels)
- Stops on labeling failure
- Fine-tunes classifier head on GPU

Colab usage:
  !pip install -q torch torchvision pillow tqdm google-genai
  !python colab_retrain_gemini_strict.py --repo-dir /content/flood-depth-estimator --version v4
"""

from __future__ import annotations

import argparse
import csv
import getpass
import re
import shutil
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


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
        state = checkpoint.get("model_state_dict", checkpoint)
        if isinstance(state, dict):
            return state
    raise RuntimeError("Checkpoint format is invalid.")


def load_base_model(model: nn.Module, checkpoint_path: Path, device: torch.device) -> Dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Base checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = _extract_state_dict(checkpoint)
    target_state = model.state_dict()

    matched = {k: v for k, v in state.items() if k in target_state and target_state[k].shape == v.shape}

    # Legacy remap support.
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
            if mapped and mapped in target_state and target_state[mapped].shape == value.shape:
                remapped[mapped] = value
        matched = remapped

    if len(matched) < 50:
        raise RuntimeError(
            f"Base model could not be loaded properly ({len(matched)} matched tensors). "
            "Ensure models/best_flood_model_water_aware.pth is the correct checkpoint."
        )

    target_state.update(matched)
    model.load_state_dict(target_state)
    print(f"Loaded base model tensors: {len(matched)}/{len(target_state)}")
    return checkpoint if isinstance(checkpoint, dict) else {}


def create_gemini_callable(api_key: str, model_name: str) -> Callable[[Path, str], str]:
    try:
        from google import genai as modern_genai  # type: ignore
        from google.genai import types as modern_types  # type: ignore

        client = modern_genai.Client(api_key=api_key)

        def call(image_path: Path, prompt: str) -> str:
            with open(image_path, "rb") as f:
                data = f.read()
            mime = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            resp = client.models.generate_content(
                model=model_name,
                contents=[prompt, modern_types.Part.from_bytes(data=data, mime_type=mime)],
            )
            text = getattr(resp, "text", None)
            if not text:
                raise RuntimeError("Gemini returned empty response.")
            return text

        return call
    except Exception:
        import google.generativeai as legacy_genai  # type: ignore

        legacy_genai.configure(api_key=api_key)
        model = legacy_genai.GenerativeModel(model_name)

        def call(image_path: Path, prompt: str) -> str:
            img = Image.open(image_path).convert("RGB")
            resp = model.generate_content([prompt, img])
            text = getattr(resp, "text", None)
            if not text:
                raise RuntimeError("Gemini returned empty response.")
            return text

        return call


def parse_depth_strict(text: str) -> float:
    match = re.search(r"(-?\d+(\.\d+)?)", text)
    if not match:
        raise RuntimeError(f"No numeric depth found in Gemini response: {text[:120]}")
    value = float(match.group(1))
    if not (0.0 <= value <= 150.0):
        raise RuntimeError(f"Depth out of expected range (0..150): {value}")
    return value


def upload_images(images_dir: Path, local_images_dir: str = "") -> List[Path]:
    images_dir.mkdir(parents=True, exist_ok=True)
    if IN_COLAB:
        print("Upload your NEW training images now.")
        uploaded = files.upload()
        for name, payload in uploaded.items():
            with open(images_dir / name, "wb") as f:
                f.write(payload)
        images = sorted([p for p in images_dir.iterdir() if is_image(p)])
    else:
        if not local_images_dir:
            raise RuntimeError("Outside Colab, pass --local-images-dir")
        src_dir = Path(local_images_dir).resolve()
        if not src_dir.exists():
            raise FileNotFoundError(f"Local images dir not found: {src_dir}")
        images = sorted([p for p in src_dir.iterdir() if is_image(p)])
        for idx, src in enumerate(images, start=1):
            dst = images_dir / f"user_{idx:05d}{src.suffix.lower()}"
            shutil.copy2(src, dst)
        images = sorted([p for p in images_dir.iterdir() if is_image(p)])

    if len(images) < 20:
        raise RuntimeError(f"Need at least 20 images; found {len(images)}")
    return images


def label_images_strict(
    images: Sequence[Path],
    labels_csv: Path,
    gemini_key: str,
    gemini_model: str,
    request_delay: float,
    retry_attempts: int,
    retry_backoff: float,
) -> None:
    call_gemini = create_gemini_callable(gemini_key, gemini_model)
    prompt = (
        "Estimate flood WATER DEPTH in centimeters from this image. "
        "Use object-waterline cues (person/car/bike/buildings/curbs) if visible. "
        "Return ONLY one numeric value in cm."
    )

    rows: List[Dict[str, str]] = []
    for image_path in tqdm(images, desc="Gemini strict labeling"):
        last_error: Optional[Exception] = None
        depth_cm: Optional[float] = None
        for attempt in range(1, retry_attempts + 1):
            try:
                text = call_gemini(image_path, prompt)
                depth_cm = parse_depth_strict(text)
                break
            except Exception as exc:
                last_error = exc
                sleep_s = retry_backoff * attempt
                print(f"[retry {attempt}/{retry_attempts}] {image_path.name}: {exc}")
                time.sleep(sleep_s)

        if depth_cm is None:
            raise RuntimeError(f"Gemini labeling failed for {image_path.name}. Last error: {last_error}")

        rows.append({"filename": image_path.name, "depth_cm": f"{depth_cm:.2f}"})
        time.sleep(request_delay)

    with open(labels_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "depth_cm"])
        writer.writeheader()
        writer.writerows(rows)


class FloodDepthDataset(Dataset):
    def __init__(self, images_dir: Path, labels_csv: Path):
        items: List[Tuple[Path, float]] = []
        with open(labels_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                p = images_dir / row["filename"]
                if p.exists():
                    items.append((p, float(row["depth_cm"])))
        if not items:
            raise RuntimeError("No valid labeled samples found.")

        self.items = items
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


def train_transfer_learning(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Dict:
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.05)

    best = {"val_loss": float("inf")}
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["target"].to(device).unsqueeze(1)
                preds = model(images)
                val_loss += criterion(preds, targets).item()
                val_mae += torch.abs((preds - targets) * 100.0).mean().item()
        val_loss /= max(1, len(val_loader))
        val_mae /= max(1, len(val_loader))

        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | val_mae={val_mae:.2f}cm")

        if val_loss < best["val_loss"]:
            best = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae,
                "training_method": "strict_gemini_transfer_learning",
            }
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict Colab Gemini retraining script")
    p.add_argument("--repo-dir", default=".")
    p.add_argument("--base-model", default="models/best_flood_model_water_aware.pth")
    p.add_argument("--local-images-dir", default="")
    p.add_argument("--gemini-api-key", default="")
    p.add_argument("--gemini-model", default="gemini-1.5-flash")
    p.add_argument("--request-delay", type=float, default=3.0)
    p.add_argument("--retry-attempts", type=int, default=5)
    p.add_argument("--retry-backoff", type=float, default=4.0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--version", default="v4")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    repo_dir = Path(args.repo_dir).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"Repo dir not found: {repo_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Repo: {repo_dir}")
    print(f"Device: {device}")

    work_dir = repo_dir / "colab_strict_run"
    images_dir = work_dir / "images"
    labels_csv = work_dir / "labels.csv"
    work_dir.mkdir(parents=True, exist_ok=True)

    images = upload_images(images_dir=images_dir, local_images_dir=args.local_images_dir)
    print(f"Images ready: {len(images)}")

    gemini_key = args.gemini_api_key.strip() or getpass.getpass("Gemini API key: ").strip()
    if not gemini_key:
        raise RuntimeError("Gemini API key is required.")

    label_images_strict(
        images=images,
        labels_csv=labels_csv,
        gemini_key=gemini_key,
        gemini_model=args.gemini_model,
        request_delay=args.request_delay,
        retry_attempts=args.retry_attempts,
        retry_backoff=args.retry_backoff,
    )
    print(f"Labels saved: {labels_csv}")

    model = build_model().to(device)
    base_info = load_base_model(model, repo_dir / args.base_model, device)
    if isinstance(base_info, dict):
        print(f"Base checkpoint epoch={base_info.get('epoch','n/a')} val_loss={base_info.get('val_loss','n/a')}")

    dataset = FloodDepthDataset(images_dir=images_dir, labels_csv=labels_csv)
    n_val = max(1, int(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best = train_transfer_learning(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.learning_rate,
    )

    models_dir = repo_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_version = models_dir / f"best_flood_model_{args.version}.pth"
    out_default = models_dir / "best_flood_model_water_aware.pth"
    torch.save(best, out_version)
    shutil.copy2(out_version, out_default)
    print(f"Saved: {out_version}")
    print(f"Updated default model: {out_default}")

    if IN_COLAB:
        files.download(str(out_version))

    print("\nNext step locally:")
    print("python app.py")


if __name__ == "__main__":
    main()
