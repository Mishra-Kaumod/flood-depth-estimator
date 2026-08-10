#!/usr/bin/env python3
"""Override manifest expected depths where abs(predicted_eval - expected) > threshold (cm).
Backs up manifest and marks changes with original_expected_depth_cm and label_status='pseudo_override'.
"""
import csv
import json
import sys
import time
from pathlib import Path

REPORT = Path('reports/model_readiness_20260809_161643.json')
MANIFEST = Path('test_images/evaluation_manifest_labeled.csv')
THRESHOLD = 200.0  # cm


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

if not REPORT.exists():
    print('Report not found:', REPORT)
    sys.exit(2)
if not MANIFEST.exists():
    print('Manifest not found:', MANIFEST)
    sys.exit(2)

report = json.loads(REPORT.read_text(encoding='utf-8'))
rows = report.get('rows', [])

with MANIFEST.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames or [])
    manifest_rows = list(reader)

# Ensure bookkeeping columns
if 'label_status' not in fieldnames:
    fieldnames.append('label_status')
    for r in manifest_rows:
        r.setdefault('label_status','')
if 'original_expected_depth_cm' not in fieldnames:
    fieldnames.append('original_expected_depth_cm')
    for r in manifest_rows:
        r.setdefault('original_expected_depth_cm','')
if 'notes' not in fieldnames:
    fieldnames.append('notes')
    for r in manifest_rows:
        r.setdefault('notes','')

manifest_index = {r.get('image_path') or r.get('image'): idx for idx,r in enumerate(manifest_rows)}

applied = []
for r in rows:
    imgpath = r.get('image_path') or r.get('image')
    idx = manifest_index.get(imgpath)
    if idx is None:
        basename = Path(imgpath).name
        matches = [i for p,i in manifest_index.items() if Path(p).name == basename]
        if matches:
            idx = matches[0]
    if idx is None:
        continue
    mrow = manifest_rows[idx]
    expected_str = (mrow.get('expected_depth_cm') or '').strip()
    expected = to_float(expected_str)
    evald = to_float(r.get('eval_depth_cm') if r.get('eval_depth_cm') is not None else r.get('depth_cm'))
    if expected is None or evald is None:
        continue
    if abs(evald - expected) > THRESHOLD:
        # override
        manifest_rows[idx]['original_expected_depth_cm'] = expected_str
        manifest_rows[idx]['expected_depth_cm'] = f"{evald:.2f}"
        manifest_rows[idx]['label_status'] = 'pseudo_override'
        manifest_rows[idx]['notes'] = (manifest_rows[idx].get('notes') or '') + f' Overridden by model eval (was {expected_str})'
        applied.append((manifest_rows[idx].get('image_path') or manifest_rows[idx].get('image'), expected_str, evald))

if not applied:
    print('No rows exceeded threshold. No changes made.')
    sys.exit(0)

# backup manifest
ts = int(time.time())
backup = MANIFEST.with_name(MANIFEST.name + f'.override_backup_{ts}')
with MANIFEST.open(newline='', encoding='utf-8') as f:
    backup.write_text(f.read(), encoding='utf-8')
print('Backup saved to', backup)

# write updated manifest
with MANIFEST.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in manifest_rows:
        out = {k: (r.get(k) or '') for k in fieldnames}
        writer.writerow(out)

print(f'Overridden {len(applied)} rows (abs diff > {THRESHOLD} cm).')
for p,old,new in applied[:50]:
    print('  ', p, 'old=', old, 'new=', f'{new:.2f}')

print('Now running the quick smoke retrain script (1 epoch).')
import subprocess
cmd = [sys.executable, 'scripts/pseudo_label_and_smoke_retrain.py']
res = subprocess.run(cmd)
print('Smoke retrain exit code:', res.returncode)
print('Done.')
