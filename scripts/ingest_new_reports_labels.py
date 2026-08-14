#!/usr/bin/env python3
"""
Ingest new labeled images from reports/new_labels/* where labels are encoded in filenames like '115cm.jpg' or '115.5cm.png'.
- Scans reports/new_labels recursively for image files.
- Extracts depth (cm) from filename patterns like 115cm, 115.5cm, 115_cm, etc.
- Copies images into training_data/images/ (creates folder if missing).
- Updates training_data/labels.csv (backs up existing file). If a filename exists in CSV, updates depth; else appends a new row.
- Prints a summary of changes and writes a small changes CSV at reports/new_labels_ingest.csv

Run: py -3 scripts\ingest_new_reports_labels.py
"""
from pathlib import Path
import re
import csv
import shutil
import sys

REPO = Path('.').resolve()
NEW_DIR = REPO / 'reports' / 'new_labels'
TRAIN_DIR = REPO / 'training_data'
TRAIN_IMAGES = TRAIN_DIR / 'images'
LABELS_CSV = TRAIN_DIR / 'labels.csv'
BACKUP = LABELS_CSV.with_name(LABELS_CSV.name + '.bak')
OUT_CHANGES = REPO / 'reports' / 'new_labels_ingest.csv'

if not NEW_DIR.exists():
    print('No new_labels directory at', NEW_DIR)
    sys.exit(1)

TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# load existing labels
labels = {}
fieldnames = ['filename','depth_cm']
if LABELS_CSV.exists():
    with LABELS_CSV.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            fname = r.get('filename')
            if not fname:
                continue
            try:
                depth = float(r.get('depth_cm'))
            except Exception:
                continue
            labels[fname] = depth

# backup existing labels
if LABELS_CSV.exists():
    shutil.copy2(LABELS_CSV, BACKUP)
    print('Backed up existing labels.csv to', BACKUP)

# pattern to find depth in filename
pattern = re.compile(r'(?P<depth>\d+(?:[\.,]\d+)?)\s*_?cm', re.IGNORECASE)
# also accept forms like 115cm, 115_cm, 115.5cm
pattern2 = re.compile(r'(?P<depth>\d+(?:[\.,]\d+)?)', re.IGNORECASE)

changes = []
count_added = 0
count_updated = 0

for img in NEW_DIR.rglob('*'):
    if not img.is_file():
        continue
    low = img.suffix.lower()
    if low not in ('.jpg','.jpeg','.png','.webp','.gif'):
        continue
    name = img.name
    m = pattern.search(name)
    depth = None
    if m:
        depth = float(m.group('depth').replace(',','.'))
    else:
        # try to find a number near end before extension
        m2 = re.search(r'([0-9]{1,4}(?:[\.,][0-9]+)?)', name)
        if m2:
            # heuristic: accept if filename contains 'cm' in parent folder name or name with 'depth'
            if 'cm' in name.lower() or 'depth' in str(img.parent).lower() or 'depth' in name.lower():
                depth = float(m2.group(1).replace(',','.'))
    if depth is None:
        # check for sidecar text file with same basename
        side_txt = img.with_suffix('.txt')
        if side_txt.exists():
            try:
                content = side_txt.read_text(encoding='utf-8').strip()
                m3 = re.search(r'(-?\d+(?:[\.,]\d+)?)', content)
                if m3:
                    depth = float(m3.group(1).replace(',','.'))
            except Exception:
                pass
    if depth is None:
        continue
    # copy image into training_data/images (avoid name collisions)
    dest = TRAIN_IMAGES / name
    i = 1
    while dest.exists():
        # if file is identical, skip copy
        try:
            if dest.stat().st_size == img.stat().st_size:
                break
        except Exception:
            pass
        dest = TRAIN_IMAGES / f"{img.stem}_{i}{img.suffix}"
        i += 1
    try:
        shutil.copy2(img, dest)
    except Exception as exc:
        print('Failed to copy', img, '->', dest, exc)
        continue
    fname = dest.name
    old = labels.get(fname)
    if old is None:
        labels[fname] = float(depth)
        changes.append({'action':'added','filename':fname,'depth_cm':f'{depth:.2f}','source':str(img)})
        count_added += 1
    else:
        if abs(old - float(depth)) > 1e-6:
            labels[fname] = float(depth)
            changes.append({'action':'updated','filename':fname,'depth_cm':f'{depth:.2f}','old_depth':f'{old:.2f}','source':str(img)})
            count_updated += 1

# write back labels csv
with LABELS_CSV.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for k,v in sorted(labels.items()):
        writer.writerow({'filename':k,'depth_cm':f'{v:.2f}'})

# write changes report
OUT_CHANGES.parent.mkdir(parents=True, exist_ok=True)
with OUT_CHANGES.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['action','filename','depth_cm','old_depth','source'])
    writer.writeheader()
    for c in changes:
        writer.writerow({
            'action': c.get('action',''),
            'filename': c.get('filename',''),
            'depth_cm': c.get('depth_cm',''),
            'old_depth': c.get('old_depth',''),
            'source': c.get('source','')
        })

print(f'Added {count_added} new labels, updated {count_updated} existing labels.')
print('Wrote updated labels CSV to', LABELS_CSV)
print('Wrote changes log to', OUT_CHANGES)
print('Training images folder:', TRAIN_IMAGES)
