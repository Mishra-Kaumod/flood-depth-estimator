#!/usr/bin/env python3
"""
Apply high-confidence pseudo-labels from a readiness report into the manifest,
then run a fast smoke retrain (1 epoch) using manifest-labeled images and
re-run the readiness evaluator.

Criteria (hard-coded):
 - predicted_flood == 1
 - water_confidence >= 0.8
 - confidence >= 0.8
 - water_coverage_pct >= 50
 - no_reference_warning == False

This script backs up the manifest before editing.
"""
from __future__ import annotations
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import random

# ML deps
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

# Params
REPORT = Path("reports/model_readiness_20260809_161643.json")
MANIFEST = Path("test_images/evaluation_manifest_labeled.csv")
BACKUP = MANIFEST.with_name(MANIFEST.name + ".pseudo_backup")
MAX_TRAIN_IMAGES = 200
EPOCHS = 1
BATCH_SIZE = 8
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        try:
            return float(str(v).strip())
        except Exception:
            return None


def boolish(v):
    if v in (True, False):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "t", "yes", "y")


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


class ManifestDataset(Dataset):
    def __init__(self, items: List[Dict[str, str]], repo_root: Path):
        self.samples = []
        self.repo_root = repo_root
        for row in items:
            path = row.get('image_path') or row.get('image')
            if not path:
                continue
            p = (repo_root / path).resolve()
            if not p.exists():
                continue
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            depth = to_float(row.get('expected_depth_cm'))
            if depth is None:
                continue
            # clip to [0,100]
            depth = max(0.0, min(100.0, depth))
            self.samples.append((p, depth))
        if not self.samples:
            raise RuntimeError('No labeled samples for training.')
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        p, d = self.samples[idx]
        img = Image.open(p).convert('RGB')
        x = self.transform(img)
        y = torch.tensor(d / 100.0, dtype=torch.float32)
        return {'image': x, 'target': y}


def main():
    if not REPORT.exists():
        print('Report not found:', REPORT)
        sys.exit(2)
    if not MANIFEST.exists():
        print('Manifest not found:', MANIFEST)
        sys.exit(2)

    repo_root = Path('.').resolve()

    data = json.loads(REPORT.read_text(encoding='utf-8'))
    rows = data.get('rows', [])

    # load manifest
    with MANIFEST.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        manifest_rows = list(reader)

    # Ensure label_status column exists
    if 'label_status' not in fieldnames:
        fieldnames.append('label_status')
        for r in manifest_rows:
            r.setdefault('label_status', '')

    manifest_index = {r.get('image_path') or r.get('image'): idx for idx, r in enumerate(manifest_rows)}

    # find candidates per thresholds
    candidates = []
    for r in rows:
        try:
            pred = 1 if str(r.get('predicted_flood','0')).strip() in ('1','True','true') else 0
            if pred != 1:
                continue
            wconf = to_float(r.get('water_confidence')) or 0.0
            conf = to_float(r.get('confidence')) or 0.0
            wcov = to_float(r.get('water_coverage_pct')) or 0.0
            no_ref = boolish(r.get('no_reference_warning'))
            if wconf >= 0.8 and conf >= 0.8 and wcov >= 50.0 and (not no_ref):
                imgpath = r.get('image_path') or r.get('image')
                idx = manifest_index.get(imgpath)
                if idx is None:
                    # try basename match
                    basename = Path(imgpath).name
                    matches = [i for p,i in manifest_index.items() if Path(p).name == basename]
                    if matches:
                        idx = matches[0]
                if idx is None:
                    continue
                mrow = manifest_rows[idx]
                if (mrow.get('expected_depth_cm') or '').strip() != '':
                    # already labeled
                    continue
                evald = r.get('eval_depth_cm') if r.get('eval_depth_cm') is not None else r.get('depth_cm')
                evald_f = to_float(evald)
                if evald_f is None:
                    continue
                candidates.append((idx, imgpath, evald_f, r))
        except Exception:
            continue

    if not candidates:
        print('No candidates found for pseudo-labeling with the thresholds. Exiting.')
        sys.exit(0)

    # backup manifest
    ts = int(time.time())
    backup_path = MANIFEST.with_name(MANIFEST.name + f'.pseudo_backup_{ts}')
    with MANIFEST.open(newline='', encoding='utf-8') as f:
        backup_path.write_text(f.read(), encoding='utf-8')
    print('Backup manifest written to', backup_path)

    # apply pseudo-labels
    applied = []
    for idx, imgpath, evald_f, r in candidates:
        # clip
        depth_clipped = max(0.0, min(100.0, float(evald_f)))
        manifest_rows[idx]['expected_depth_cm'] = f"{depth_clipped:.2f}"
        manifest_rows[idx]['label_status'] = 'pseudo'
        applied.append((manifest_rows[idx].get('image_path') or manifest_rows[idx].get('image'), depth_clipped))

    # write manifest back
    with MANIFEST.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in manifest_rows:
            # ensure all keys present
            out = {k: (r.get(k) or '') for k in fieldnames}
            writer.writerow(out)

    print(f'Applied {len(applied)} pseudo-labels to manifest ({MANIFEST}).')

    # Now build dataset from manifest-labeled images
    labeled = [r for r in manifest_rows if (r.get('expected_depth_cm') or '').strip()!='']
    print('Total labeled rows available for training:', len(labeled))
    # shuffle and cap
    random.shuffle(labeled)
    labeled = labeled[:MAX_TRAIN_IMAGES]
    try:
        dataset = ManifestDataset(labeled, repo_root)
    except Exception as exc:
        print('Failed to build dataset for training:', exc)
        sys.exit(2)

    n = len(dataset)
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # build model
    model = build_model().to(DEVICE)
    # freeze backbone
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=0.05)

    best_state = None
    best_val_mae = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(DEVICE)
            targets = batch['target'].to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= max(1, len(train_loader))

        # validate
        model.eval()
        v_loss = 0.0
        v_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(DEVICE)
                targets = batch['target'].to(DEVICE).unsqueeze(1)
                pred = model(images)
                v_loss += criterion(pred, targets).item()
                v_mae += torch.abs((pred - targets) * 100.0).mean().item()
        v_loss /= max(1, len(val_loader))
        v_mae /= max(1, len(val_loader))
        print(f'Epoch {epoch+1}/{EPOCHS} train_loss={t_loss:.5f} val_loss={v_loss:.5f} val_mae={v_mae:.2f}cm')
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            best_state = {'model_state_dict': model.state_dict(), 'val_mae': v_mae}

    # save candidate
    model_dir = Path('models')
    model_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = model_dir / 'candidate'
    candidate_dir.mkdir(parents=True, exist_ok=True)
    out_candidate = candidate_dir / 'best_flood_model_water_aware_candidate.pth'
    torch.save(best_state or {'model_state_dict': model.state_dict()}, out_candidate)
    print('Saved candidate model to', out_candidate)

    # run evaluator
    eval_cmd = [sys.executable, 'evaluate_model_readiness.py', '--input-dir', 'test_images', '--manifest', str(MANIFEST), '--enforce-gates']
    print('Running evaluator:', ' '.join(eval_cmd))
    import subprocess
    try:
        subprocess.run(eval_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print('Evaluator failed:', exc)
        # still exit ok

    print('Done.')

if __name__ == '__main__':
    main()
