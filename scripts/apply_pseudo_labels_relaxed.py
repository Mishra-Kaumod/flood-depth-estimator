#!/usr/bin/env python3
"""Apply pseudo-labels with relaxed thresholds (wconf>=0.6, conf>=0.6, wcov>=50, no_reference_warning==False)
Updates test_images/evaluation_manifest_labeled.csv with expected_depth_cm and label_status='pseudo'.
Backs up manifest before editing.
"""
import csv
import json
import sys
import time
from pathlib import Path

REPORT = Path('reports/model_readiness_20260809_161643.json')
MANIFEST = Path('test_images/evaluation_manifest_labeled.csv')


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
    return s in ('1','true','t','yes','y')

if not REPORT.exists():
    print('Report not found:', REPORT)
    sys.exit(2)
if not MANIFEST.exists():
    print('Manifest not found:', MANIFEST)
    sys.exit(2)

data = json.loads(REPORT.read_text(encoding='utf-8'))
rows = data.get('rows', [])

with MANIFEST.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames or [])
    manifest_rows = list(reader)

if 'label_status' not in fieldnames:
    fieldnames.append('label_status')
    for r in manifest_rows:
        r.setdefault('label_status','')

manifest_index = {r.get('image_path') or r.get('image'): idx for idx,r in enumerate(manifest_rows)}

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
        if wconf >= 0.6 and conf >= 0.6 and wcov >= 50.0 and (not no_ref):
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
            if (mrow.get('expected_depth_cm') or '').strip() != '':
                continue
            evald = r.get('eval_depth_cm') if r.get('eval_depth_cm') is not None else r.get('depth_cm')
            evald_f = to_float(evald)
            if evald_f is None:
                continue
            candidates.append((idx,imgpath,evald_f))
    except Exception:
        continue

if not candidates:
    print('No candidates found with relaxed thresholds. Exiting.')
    sys.exit(0)

# backup
ts = int(time.time())
backup_path = MANIFEST.with_name(MANIFEST.name + f'.relaxed_backup_{ts}')
with MANIFEST.open(newline='', encoding='utf-8') as f:
    backup_path.write_text(f.read(), encoding='utf-8')
print('Backed up manifest to', backup_path)

applied = []
for idx,imgpath,evald_f in candidates:
    depth_clipped = max(0.0, min(100.0, float(evald_f)))
    manifest_rows[idx]['expected_depth_cm'] = f"{depth_clipped:.2f}"
    manifest_rows[idx]['label_status'] = 'pseudo'
    applied.append((manifest_rows[idx].get('image_path') or manifest_rows[idx].get('image'), depth_clipped))

with MANIFEST.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in manifest_rows:
        out = {k: (r.get(k) or '') for k in fieldnames}
        writer.writerow(out)

print(f'Applied {len(applied)} pseudo-labels to manifest.')
for p,d in applied[:30]:
    print('  ', p, '->', d)
print('Done.')
