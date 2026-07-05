"""
One-go Colab executable: enterprise Option A retraining pipeline.

Pipeline:
  - User uploads images (optional but recommended)
  - Download one compact Kaggle dataset with flood + non-flood classes
  - Object-aware Gemini pseudo-labeling (no manual labels required)
  - GPU transfer-learning fine-tuning
  - Download trained .pth model

Usage in Google Colab:
  !pip install -q torch torchvision google-genai pillow tqdm kaggle
  !python Retrain_Existing_Model_GitHub.py --version v3 --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import logging
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(items, **_kwargs):  # type: ignore
        return items

try:
    from google.colab import files  # type: ignore

    IN_COLAB = True
except Exception:
    IN_COLAB = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("one_go_retrain")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
REFERENCE_OBJECTS = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
}


def run_cmd(command: Sequence[str], cwd: Optional[Path] = None, check: bool = True) -> subprocess.CompletedProcess:
    logger.info("Running: %s", " ".join(command))
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=check,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def discover_images(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if is_image_file(p)])


def clone_or_prepare_repo(repo_url: str, repo_dir: str) -> Path:
    repo_path = Path(repo_dir)
    if repo_path.exists():
        logger.info("Repo already exists at %s, reusing it", repo_path)
    else:
        run_cmd(["git", "clone", repo_url, repo_dir])
    run_cmd(["git", "lfs", "install"], cwd=repo_path)
    run_cmd(["git", "lfs", "pull"], cwd=repo_path)
    return repo_path


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


def load_existing_model(repo_path: Path, model_rel_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    ckpt_path = repo_path / model_rel_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model = build_model().to(device)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        logger.warning("Strict state_dict load failed, falling back to strict=False")
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    logger.info("Loaded base model from %s", ckpt_path)
    if isinstance(checkpoint, dict):
        logger.info(
            "Base metrics: epoch=%s val_loss=%s val_mae=%s",
            checkpoint.get("epoch", "n/a"),
            checkpoint.get("val_loss", "n/a"),
            checkpoint.get("val_mae", "n/a"),
        )
    return model, checkpoint if isinstance(checkpoint, dict) else {}


@dataclass
class GeminiClient:
    mode: str
    client: object
    model_name: str
    genai_types: Optional[object] = None


def create_gemini_client(api_key: str, model_name: str) -> GeminiClient:
    try:
        from google import genai as modern_genai  # type: ignore
        from google.genai import types as modern_types  # type: ignore

        client = modern_genai.Client(api_key=api_key)
        return GeminiClient(mode="modern", client=client, model_name=model_name, genai_types=modern_types)
    except Exception:
        import google.generativeai as legacy_genai  # type: ignore

        legacy_genai.configure(api_key=api_key)
        return GeminiClient(mode="legacy", client=legacy_genai, model_name=model_name)


def parse_depth_cm(response_text: str, default_depth: float) -> float:
    match = re.search(r"(-?\d+(\.\d+)?)", response_text)
    if not match:
        return default_depth
    value = float(match.group(1))
    return float(np.clip(value, 0.0, 100.0))


def create_object_detector(device: torch.device) -> Tuple[nn.Module, List[str]]:
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights).to(device)
    model.eval()
    categories: List[str] = list(weights.meta.get("categories", []))
    return model, categories


def detect_reference_objects(
    detector: nn.Module,
    categories: Sequence[str],
    image_path: Path,
    device: torch.device,
    score_threshold: float = 0.55,
    max_objects: int = 5,
) -> List[str]:
    from torchvision.transforms.functional import to_tensor

    image = Image.open(image_path).convert("RGB")
    tensor = to_tensor(image).to(device)

    with torch.no_grad():
        output = detector([tensor])[0]

    labels = output.get("labels", torch.empty(0, dtype=torch.long))
    scores = output.get("scores", torch.empty(0))
    found: List[str] = []
    for label, score in zip(labels.tolist(), scores.tolist()):
        if score < score_threshold:
            continue
        name = categories[label] if 0 <= label < len(categories) else f"class_{label}"
        if name in REFERENCE_OBJECTS and name not in found:
            found.append(name)
        if len(found) >= max_objects:
            break
    return found


def label_image_with_gemini(
    gemini: GeminiClient,
    image_path: Path,
    object_cues: Sequence[str],
    default_depth: float,
    pause_seconds: float,
) -> float:
    cues = ", ".join(object_cues) if object_cues else "none detected"
    prompt = f"""
Estimate flood WATER DEPTH in centimeters from this image.

Detected reference objects from computer vision: {cues}

Rules:
1. If no flooding is visible, return 0.
2. Use object-waterline cues where available:
   - person: ankle ~10cm, knee ~45cm, waist ~90cm
   - car/bike: tire ~30cm, bumper ~45-55cm
3. If no reliable object cue exists, estimate using road coverage and waterline context.
4. Return ONLY one numeric value in cm.
"""

    try:
        if gemini.mode == "modern":
            assert gemini.genai_types is not None
            with open(image_path, "rb") as f:
                payload = f.read()
            mime_type = "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
            response = gemini.client.models.generate_content(  # type: ignore[attr-defined]
                model=gemini.model_name,
                contents=[
                    prompt,
                    gemini.genai_types.Part.from_bytes(data=payload, mime_type=mime_type),
                ],
            )
            text = getattr(response, "text", "") or ""
        else:
            pil_img = Image.open(image_path).convert("RGB")
            model = gemini.client.GenerativeModel(gemini.model_name)  # type: ignore[attr-defined]
            response = model.generate_content([prompt, pil_img])
            text = getattr(response, "text", "") or ""

        depth_cm = parse_depth_cm(text, default_depth=default_depth)
        time.sleep(pause_seconds)
        return depth_cm
    except Exception as exc:
        logger.warning("Gemini failed for %s: %s (fallback %.1fcm)", image_path.name, exc, default_depth)
        return default_depth


def label_images_to_csv(
    images: List[Path],
    csv_path: Path,
    gemini_api_key: str,
    gemini_model: str,
    default_depth: float,
    pause_seconds: float,
    detector: nn.Module,
    categories: Sequence[str],
    detector_device: torch.device,
    object_score_threshold: float,
) -> None:
    gemini = create_gemini_client(gemini_api_key, gemini_model)
    rows: List[Dict[str, str]] = []
    for image_path in tqdm(images, desc="Gemini labeling"):
        object_cues = detect_reference_objects(
            detector=detector,
            categories=categories,
            image_path=image_path,
            device=detector_device,
            score_threshold=object_score_threshold,
        )
        depth_cm = label_image_with_gemini(
            gemini,
            image_path,
            object_cues=object_cues,
            default_depth=default_depth,
            pause_seconds=pause_seconds,
        )
        rows.append(
            {
                "filename": image_path.name,
                "depth_cm": f"{depth_cm:.2f}",
                "objects": ";".join(object_cues),
                "timestamp": datetime.now().isoformat(),
            }
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "depth_cm", "objects", "timestamp"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved labels CSV: %s (%d rows)", csv_path, len(rows))


def _write_kaggle_json(kaggle_json_path: Path, username: str, key: str) -> None:
    kaggle_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"username": username, "key": key}
    with open(kaggle_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.chmod(kaggle_json_path, 0o600)
    logger.info("Saved Kaggle credentials to %s", kaggle_json_path)


def setup_kaggle_credentials_if_needed(kaggle_username: str = "", kaggle_key: str = "") -> None:
    kaggle_json_path = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json_path.exists():
        return

    username = kaggle_username.strip()
    key = kaggle_key.strip()

    if username and key:
        _write_kaggle_json(kaggle_json_path, username=username, key=key)
        return

    if username and not key:
        raise RuntimeError("Kaggle username provided but key is missing. Pass --kaggle-key too.")
    if key and not username:
        raise RuntimeError("Kaggle key provided but username is missing. Pass --kaggle-username too.")

    logger.info("Kaggle credentials not found.")
    logger.info("Option 1 (recommended): paste Kaggle username + API key")
    logger.info("Option 2: upload kaggle.json")

    pasted_username = input("Kaggle username (press Enter to skip and upload kaggle.json): ").strip()
    if pasted_username:
        pasted_key = getpass.getpass("Kaggle API key: ").strip()
        if not pasted_key:
            raise RuntimeError("Kaggle API key cannot be empty.")
        _write_kaggle_json(kaggle_json_path, username=pasted_username, key=pasted_key)
        return

    if not IN_COLAB:
        raise RuntimeError(
            "kaggle.json not found and no pasted credentials provided. "
            "Pass --kaggle-username and --kaggle-key, or place ~/.kaggle/kaggle.json."
        )

    logger.info("Upload kaggle.json from your Kaggle account settings")
    uploaded = files.upload()
    if "kaggle.json" not in uploaded:
        raise RuntimeError("Upload failed: kaggle.json is required")

    kaggle_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(kaggle_json_path, "wb") as f:
        f.write(uploaded["kaggle.json"])
    os.chmod(kaggle_json_path, 0o600)
    logger.info("Saved Kaggle credentials to %s", kaggle_json_path)


def download_kaggle_datasets(dataset_slugs: List[str], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    for slug in dataset_slugs:
        slug = slug.strip()
        if not slug:
            continue
        target = out_dir / slug.replace("/", "_")
        target.mkdir(parents=True, exist_ok=True)
        try:
            run_cmd(
                ["kaggle", "datasets", "download", "-d", slug, "-p", str(target), "--unzip"],
                check=True,
            )
            downloaded.append(target)
            logger.info("Downloaded Kaggle dataset: %s", slug)
        except subprocess.CalledProcessError as exc:
            logger.warning("Kaggle download failed for %s: %s", slug, exc.stderr.strip())
    return downloaded


def _bucket_for_path(path: Path) -> str:
    text = str(path.parent).lower()
    if re.search(r"non[-_ ]?flood|no[-_ ]?flood|dry|normal|clear", text):
        return "non_flooded"
    if re.search(r"flood|waterlog|inundat|submerg", text):
        return "flooded"
    return "other"


def select_balanced_kaggle_images(
    kaggle_dirs: Sequence[Path],
    max_kaggle_images: int,
    seed: int,
) -> List[Path]:
    buckets: Dict[str, List[Path]] = {"flooded": [], "non_flooded": [], "other": []}
    for root in kaggle_dirs:
        for path in discover_images(root):
            buckets[_bucket_for_path(path)].append(path)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    if max_kaggle_images <= 0:
        return buckets["flooded"] + buckets["non_flooded"] + buckets["other"]

    half = max_kaggle_images // 2
    selected: List[Path] = []
    selected.extend(buckets["flooded"][:half])
    selected.extend(buckets["non_flooded"][:half])

    remaining = max_kaggle_images - len(selected)
    if remaining > 0:
        tail_pool = buckets["other"] + buckets["flooded"][half:] + buckets["non_flooded"][half:]
        selected.extend(tail_pool[:remaining])

    return selected


def upload_user_images_colab(out_dir: Path) -> List[Path]:
    if not IN_COLAB:
        return discover_images(out_dir)
    logger.info("Upload your flood images now (jpg/png/webp)")
    uploaded = files.upload()
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in uploaded.items():
        with open(out_dir / filename, "wb") as f:
            f.write(data)
    return discover_images(out_dir)


def materialize_training_images(
    user_images: Sequence[Path],
    kaggle_images: Sequence[Path],
    merged_dir: Path,
) -> List[Path]:
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_paths: List[Path] = []
    counter = 0

    for src in list(user_images) + list(kaggle_images):
        if not src.exists():
            continue
        counter += 1
        source_prefix = "user" if src in user_images else "kaggle"
        dst_name = f"{source_prefix}_{counter:06d}{src.suffix.lower()}"
        dst = merged_dir / dst_name
        shutil.copy2(src, dst)
        merged_paths.append(dst)
    return merged_paths


class LabeledDepthDataset(Dataset):
    def __init__(self, image_dir: Path, labels_csv: Path):
        self.records: List[Tuple[Path, float]] = []
        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_path = image_dir / row["filename"]
                if not image_path.exists():
                    continue
                depth_cm = float(row["depth_cm"])
                depth_cm = float(np.clip(depth_cm, 0.0, 100.0))
                self.records.append((image_path, depth_cm))

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        if not self.records:
            raise RuntimeError("No labeled images found for training.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path, depth_cm = self.records[idx]
        img = Image.open(path).convert("RGB")
        tensor = self.transform(img)
        target_norm = torch.tensor(depth_cm / 100.0, dtype=torch.float32)
        return {"image": tensor, "target_norm": target_norm}


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 0.05):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        residual = torch.abs(pred - target)
        small = 0.5 * residual.pow(2)
        large = self.delta * (residual - 0.5 * self.delta)
        return torch.where(residual < self.delta, small, large).mean()


def configure_trainable_params(model: nn.Module, freeze_backbone: bool) -> None:
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = True


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    freeze_backbone: bool,
) -> Dict:
    configure_trainable_params(model, freeze_backbone=freeze_backbone)
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info("Trainable params: %s", f"{sum(p.numel() for p in trainable):,}")

    optimizer = optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = HuberLoss(delta=0.05)

    best_val_loss = float("inf")
    best_checkpoint: Dict = {}
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_mae_cm": [], "val_mae_cm": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_mae_cm = 0.0
        for batch in train_loader:
            images = batch["image"].to(device)
            targets = batch["target_norm"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_mae_cm += torch.abs((preds - targets) * 100.0).mean().item()

        train_loss /= max(1, len(train_loader))
        train_mae_cm /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_mae_cm = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                targets = batch["target_norm"].to(device).unsqueeze(1)
                preds = model(images)
                loss = criterion(preds, targets)

                val_loss += loss.item()
                val_mae_cm += torch.abs((preds - targets) * 100.0).mean().item()

        val_loss /= max(1, len(val_loader))
        val_mae_cm /= max(1, len(val_loader))
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae_cm"].append(train_mae_cm)
        history["val_mae_cm"].append(val_mae_cm)

        logger.info(
            "Epoch %d/%d | train_loss=%.5f train_mae=%.2fcm | val_loss=%.5f val_mae=%.2fcm",
            epoch + 1,
            epochs,
            train_loss,
            train_mae_cm,
            val_loss,
            val_mae_cm,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_mae": val_mae_cm,
                "metrics": history,
                "training_method": "transfer_learning_frozen_backbone" if freeze_backbone else "full_finetune",
            }

    if not best_checkpoint:
        raise RuntimeError("Training did not produce a checkpoint")
    return best_checkpoint


def update_model_registry(registry_path: Path, version: str, metadata: Dict) -> None:
    registry: Dict = {}
    if registry_path.exists():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
                if not isinstance(registry, dict):
                    registry = {}
        except Exception:
            registry = {}

    registry[version] = metadata
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def save_outputs(repo_path: Path, checkpoint: Dict, version: str, update_default: bool) -> Tuple[Path, Optional[Path]]:
    models_dir = repo_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    versioned = models_dir / f"best_flood_model_{version}.pth"
    torch.save(checkpoint, versioned)

    default_path: Optional[Path] = None
    if update_default:
        default_path = models_dir / "best_flood_model_water_aware.pth"
        shutil.copy2(versioned, default_path)
    return versioned, default_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-go Colab retraining script")
    parser.add_argument("--repo-url", default="https://github.com/Mishra-Kaumod/flood-depth-estimator.git")
    parser.add_argument("--repo-dir", default="flood-depth-estimator")
    parser.add_argument("--model-path", default="models/best_flood_model_water_aware.pth")
    parser.add_argument(
        "--kaggle-datasets",
        default="jannalipka/flood-detection-image-dataset",
        help="Comma-separated Kaggle dataset slugs. Default is a compact flood/non-flood dataset.",
    )
    parser.add_argument("--kaggle-username", default="")
    parser.add_argument("--kaggle-key", default="")
    parser.add_argument("--upload-dir", default="", help="Optional local path for user images when not in Colab")
    parser.add_argument("--gemini-model", default="gemini-1.5-flash")
    parser.add_argument("--gemini-api-key", default="")
    parser.add_argument("--default-depth-cm", type=float, default=30.0)
    parser.add_argument("--gemini-pause-seconds", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--unfreeze-backbone", action="store_true")
    parser.add_argument("--version", default="v3")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--max-kaggle-images", type=int, default=600)
    parser.add_argument("--object-score-threshold", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-update-default", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("FLOOD DEPTH RETRAINING (ONE-GO EXECUTABLE)")
    print(f"Mode: Option A (Upload + Kaggle) | Device: {device}")
    print("=" * 78)

    repo_path = clone_or_prepare_repo(args.repo_url, args.repo_dir)
    model, base_checkpoint = load_existing_model(repo_path, args.model_path, device)

    workspace = Path("one_go_workspace")
    user_dir = workspace / "user_images"
    kaggle_root = workspace / "kaggle_raw"
    merged_dir = workspace / "training_images"
    workspace.mkdir(exist_ok=True)

    user_images: List[Path] = []
    if IN_COLAB:
        user_images = upload_user_images_colab(user_dir)
        logger.info("User uploaded images: %d", len(user_images))
    elif args.upload_dir:
        user_images = discover_images(Path(args.upload_dir))
        logger.info("Local user images: %d", len(user_images))
    else:
        logger.info("No user upload dir provided outside Colab; continuing with Kaggle images only.")

    setup_kaggle_credentials_if_needed(
        kaggle_username=args.kaggle_username,
        kaggle_key=args.kaggle_key,
    )
    slugs = [s.strip() for s in args.kaggle_datasets.split(",") if s.strip()]
    kaggle_dirs = download_kaggle_datasets(slugs, kaggle_root)
    kaggle_images = select_balanced_kaggle_images(
        kaggle_dirs=kaggle_dirs,
        max_kaggle_images=args.max_kaggle_images,
        seed=args.seed,
    )
    logger.info("Selected Kaggle images (balanced/capped): %d", len(kaggle_images))

    merged_images = materialize_training_images(user_images=user_images, kaggle_images=kaggle_images, merged_dir=merged_dir)
    if args.max_images > 0:
        merged_images = merged_images[: args.max_images]
    if len(merged_images) < 10:
        raise RuntimeError(f"Need at least 10 images for training, found {len(merged_images)}")
    logger.info("Total merged images for labeling/training: %d", len(merged_images))

    gemini_key = args.gemini_api_key.strip() or getpass.getpass("Enter Gemini API key: ").strip()
    labels_csv = workspace / "gemini_labels.csv"
    detector_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector, categories = create_object_detector(detector_device)
    label_images_to_csv(
        images=merged_images,
        csv_path=labels_csv,
        gemini_api_key=gemini_key,
        gemini_model=args.gemini_model,
        default_depth=args.default_depth_cm,
        pause_seconds=args.gemini_pause_seconds,
        detector=detector,
        categories=categories,
        detector_device=detector_device,
        object_score_threshold=args.object_score_threshold,
    )
    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = LabeledDepthDataset(merged_dir, labels_csv)
    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise RuntimeError("Not enough labeled samples after split")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    logger.info("Train/Val split: %d / %d", train_size, val_size)

    best_checkpoint = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        freeze_backbone=not args.unfreeze_backbone,
    )

    versioned_model_path, default_model_path = save_outputs(
        repo_path=repo_path,
        checkpoint=best_checkpoint,
        version=args.version,
        update_default=not args.no_update_default,
    )

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "version": args.version,
        "base_model_path": args.model_path,
        "base_epoch": base_checkpoint.get("epoch", "n/a"),
        "base_val_loss": base_checkpoint.get("val_loss", "n/a"),
        "new_val_loss": best_checkpoint["val_loss"],
        "new_val_mae_cm": best_checkpoint["val_mae"],
        "images_used": len(merged_images),
        "option": "A",
        "user_images": len(user_images),
        "kaggle_images": len(kaggle_images),
        "kaggle_datasets": slugs,
    }
    update_model_registry(repo_path / "model_registry.json", args.version, metadata)

    if IN_COLAB:
        logger.info("Downloading model to your machine: %s", versioned_model_path)
        files.download(str(versioned_model_path))

    print("\n" + "=" * 78)
    print("DONE: model trained and saved")
    print(f"Versioned model : {versioned_model_path}")
    if default_model_path:
        print(f"Default model   : {default_model_path}")
    print("=" * 78)
    print("Integration commands (run in local repo):")
    print(f"  git add models/best_flood_model_{args.version}.pth")
    print("  git add models/best_flood_model_water_aware.pth")
    print("  git add model_registry.json")
    print(f"  git commit -m \"Train model {args.version}: val_mae={best_checkpoint['val_mae']:.2f}cm\"")
    print("  git push origin main")
    print("  python app.py")
    print("=" * 78)


if __name__ == "__main__":
    main()
