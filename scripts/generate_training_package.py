#!/usr/bin/env python3
"""Generate a training package folder using the latest readiness report.
Creates training_data/images/ and training_data/labels.csv with columns: filename,depth_cm
Uses eval_depth_cm if present, else depth_cm.
"""
import json
from pathlib import Path
import shutil
import sys

repo_root = Path('.').resolve()
reports_dir = repo_root / 'reports'
if not reports_dir.exists():
    print('reports/ directory not found')
    sys.exit(1)

# find latest model_readiness_*.json
reports = sorted(reports_dir.glob('model_readiness_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
if not reports:
    print('No model_readiness_*.json files found in reports/')
    sys.exit(1)
report_path = reports[0]
print('Using report:', report_path.name)

data = json.loads(report_path.read_text(encoding='utf-8'))
rows = data.get('rows', [])

out_dir = repo_root / 'training_data'
images_dir = out_dir / 'images'
images_dir.mkdir(parents=True, exist_ok=True)
labels_csv = out_dir / 'labels.csv'

count = 0
with labels_csv.open('w', encoding='utf-8', newline='') as f:
    f.write('filename,depth_cm\n')
    for r in rows:
        # pick eval_depth_cm if available
        evald = r.get('eval_depth_cm') if r.get('eval_depth_cm') is not None else r.get('depth_cm')
        if evald is None:
            continue
        try:
            depth = float(evald)
        except Exception:
            try:
                depth = float(str(evald).strip())
            except Exception:
                continue
        # find image path
        img_path = r.get('image_path') or r.get('image')
        if not img_path:
            continue
        p = Path(img_path)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        if not p.exists():
            # try relative to test_images
            p2 = (repo_root / 'test_images' / Path(img_path).name).resolve()
            if p2.exists():
                p = p2
            else:
                # skip missing
                continue
        dest = images_dir / p.name
        try:
            shutil.copy2(p, dest)
        except Exception as exc:
            print('Failed to copy', p, '->', dest, exc)
            continue
        f.write(f'{dest.name},{depth:.2f}\n')
        count += 1

print(f'Wrote {count} images and labels to {out_dir}')
print('Images folder:', images_dir)
print('Labels CSV:', labels_csv)
