#!/usr/bin/env python3
"""
Validate training labels (training_data/labels.csv) vs manifest (if present).
Outputs:
 - reports/label_validation_details.csv (filename,label_depth,manifest_expected,original_expected,abs_diff_cm,band)
 - reports/label_validation.html (inline image gallery showing label vs manifest)
Run: py -3 scripts\validate_training_labels.py
"""
from pathlib import Path
import csv
import json
import io
import base64
from PIL import Image
import statistics

REPO = Path('.').resolve()
LABEL_CSV = REPO / 'training_data' / 'labels.csv'
IMAGES_DIR = REPO / 'training_data' / 'images'
MANIFEST = REPO / 'test_images' / 'evaluation_manifest_labeled.csv'
OUT_CSV = REPO / 'reports' / 'label_validation_details.csv'
OUT_HTML = REPO / 'reports' / 'label_validation.html'

BANDS = [(0,20,'0_20'), (20,50,'20_50'), (50,80,'50_80'), (80,99999,'80_plus')]

def to_float(v):
    if v is None or v=='' :
        return None
    try:
        return float(v)
    except:
        try:
            return float(str(v).strip())
        except:
            return None

# load manifest rows into dict by basename and by path
manifest_by_base = {}
manifest_by_path = {}
if MANIFEST.exists():
    with MANIFEST.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            key_path = (r.get('image_path') or r.get('image') or '').strip()
            if key_path:
                manifest_by_path[key_path] = r
            name = Path(key_path).name if key_path else ''
            if name:
                manifest_by_base[name] = r

# read labels
labels = []
with LABEL_CSV.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        filename = r.get('filename') or r.get('image') or r.get('name')
        depth = to_float(r.get('depth_cm') or r.get('expected_depth_cm') or r.get('label_depth'))
        if not filename or depth is None:
            continue
        labels.append({'filename': filename, 'depth': depth})

if not labels:
    print('No labels found in', LABEL_CSV)
    raise SystemExit(1)

rows_out = []
errs = []
band_stats = {b[2]: [] for b in BANDS}
abs_list = []
counts = 0

for item in labels:
    fname = item['filename']
    lab = item['depth']
    manifest_row = manifest_by_path.get(str(Path('test_images') / fname)) or manifest_by_base.get(fname)
    manifest_expected = None
    original_expected = None
    if manifest_row:
        manifest_expected = to_float(manifest_row.get('expected_depth_cm'))
        original_expected = to_float(manifest_row.get('original_expected_depth_cm') or manifest_row.get('original_expected'))
    abs_diff = None
    if manifest_expected is not None:
        abs_diff = abs(lab - manifest_expected)
        abs_list.append(abs_diff)
        # band by manifest_expected (if present) else by label
        exp = manifest_expected
    else:
        # band by label
        exp = lab
    band_name = None
    for lo,hi,name in BANDS:
        if lo <= exp < hi:
            band_name = name
            break
    if abs_diff is not None:
        band_stats[band_name].append(abs_diff)
    rows_out.append({'filename':fname,'label_depth_cm':f'{lab:.2f}','manifest_expected_cm':('' if manifest_expected is None else f'{manifest_expected:.2f}'),'original_expected_cm':('' if original_expected is None else f'{original_expected:.2f}'),'abs_diff_cm':('' if abs_diff is None else f'{abs_diff:.2f}'),'band':band_name or ''})

# write CSV
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['filename','label_depth_cm','manifest_expected_cm','original_expected_cm','abs_diff_cm','band'])
    writer.writeheader()
    for r in rows_out:
        writer.writerow(r)

# overall stats
summary = {}
if abs_list:
    summary['count'] = len(abs_list)
    summary['mae'] = statistics.mean(abs_list)
    summary['median_ae'] = statistics.median(abs_list)
    summary['stdev_ae'] = statistics.pstdev(abs_list) if len(abs_list)>1 else 0.0
    for lo,hi,name in BANDS:
        vals = band_stats[name]
        summary[f'{name}_count'] = len(vals)
        summary[f'{name}_mae'] = (statistics.mean(vals) if vals else None)

    # thresholds
    for thr in (10,25,50,100,200):
        within = sum(1 for v in abs_list if v <= thr)
        summary[f'within_{thr}cm_pct'] = 100.0 * within / len(abs_list)
else:
    summary['count'] = 0

# generate HTML gallery with inline thumbnails
html_parts = []
html_parts.append('<!doctype html>')
html_parts.append('<html><head><meta charset="utf-8"><title>Label validation</title>')
html_parts.append('<style>body{font-family:Arial,Helvetica,sans-serif;padding:12px;background:#fff} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));grid-gap:12px} .card{border:1px solid #ddd;padding:8px;border-radius:6px;background:#f9fbff} img{max-width:100%;height:auto;display:block;margin:0 auto 6px} .meta{font-size:13px;color:#003366} .k{font-weight:600;color:#0b5394} .warn{background:#ffecec;border-color:#ffb3b3}</style>')
html_parts.append('</head><body>')
html_parts.append('<h2 style="color:#0b5394">Label validation — generated from training_data/labels.csv</h2>')
html_parts.append('<p>Summary: ' + json.dumps(summary) + '</p>')
html_parts.append('<div class="grid">')

for r in rows_out:
    fname = r['filename']
    p = (IMAGES_DIR / fname).resolve()
    img_data = None
    if p.exists():
        try:
            im = Image.open(p).convert('RGB')
            im.thumbnail((640,480))
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=75)
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            img_data = f'data:image/jpeg;base64,{b64}'
        except Exception:
            img_data = None
    # highlight if abs_diff > 50
    try:
        ad = float(r['abs_diff_cm']) if r['abs_diff_cm']!='' else None
    except:
        ad = None
    warn = ad is not None and ad > 50
    card = []
    card.append(f'<div class="card{(" warn" if warn else "")}">')
    if img_data:
        card.append(f'<img src="{img_data}" alt="{fname}">')
    else:
        card.append(f'<div style="height:160px;background:#eee;display:flex;align-items:center;justify-content:center;color:#666">Missing: {fname}</div>')
    card.append('<div class="meta">')
    card.append(f'<span class="k">file:</span> {fname}<br/>')
    card.append(f'<span class="k">label_depth_cm:</span> {r["label_depth_cm"]} &nbsp; <span class="k">manifest_expected_cm:</span> {r["manifest_expected_cm"]}<br/>')
    card.append(f'<span class="k">abs_diff_cm:</span> {r["abs_diff_cm"]} &nbsp; <span class="k">band:</span> {r["band"]}<br/>')
    card.append('</div></div>')
    html_parts.append('\n'.join(card))

html_parts.append('</div>')
html_parts.append('<p style="font-size:12px;color:#333">CSV details: ' + str(OUT_CSV) + '</p>')
html_parts.append('</body></html>')

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
OUT_HTML.write_text('\n'.join(html_parts), encoding='utf-8')

print('Wrote CSV:', OUT_CSV)
print('Wrote HTML:', OUT_HTML)
print('Summary:')
print(json.dumps(summary, indent=2))
print('Done.')
