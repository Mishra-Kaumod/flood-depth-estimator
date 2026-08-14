#!/usr/bin/env python3
"""Generate an inline (base64) HTML gallery for the top-N depth-error images from a readiness report JSON.
Writes reports/top20_errors_gallery_inline.html with embedded images so the page is viewable without local file access permissions.
"""
import json
import sys
import base64
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('report', help='Path to readiness JSON report')
parser.add_argument('--top', type=int, default=20, help='Top N images')
parser.add_argument('--out', help='Output HTML path', default='reports/top20_errors_gallery_inline.html')
args = parser.parse_args()
report_path = Path(args.report)
if not report_path.exists():
    print('Report not found:', report_path)
    sys.exit(2)

data = json.loads(report_path.read_text(encoding='utf-8'))
rows = data.get('rows', [])

# Helper to get eval depth
def to_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except:
        try:
            return float(str(v).strip())
        except:
            return None

# Build list of (abs_error, row)
errors = []
for r in rows:
    exp = to_float(r.get('expected_depth_cm'))
    ev = to_float(r.get('eval_depth_cm') or r.get('depth_cm'))
    if exp is None or ev is None:
        continue
    ae = abs(ev - exp)
    errors.append((ae, r))

errors.sort(key=lambda x: x[0], reverse=True)
selected = [r for _,r in errors[:args.top]]

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)

html = []
html.append('<!doctype html>')
html.append('<html><head><meta charset="utf-8"><title>Top {} Depth Errors (Inline)</title>'.format(args.top))
html.append('<style>body{font-family:Arial,Helvetica,sans-serif;padding:12px;background:#fff} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));grid-gap:12px} .card{border:1px solid #ddd;padding:8px;border-radius:6px;background:#f9fbff} img{max-width:100%;height:auto;display:block;margin:0 auto 6px} .meta{font-size:13px;color:#003366} .k{font-weight:600;color:#0b5394}</style>')
html.append('</head><body>')
html.append('<h2 style="color:#0b5394">Top {} depth absolute errors — inline images</h2>'.format(args.top))
html.append('<div class="grid">')

for r in selected:
    img_path = r.get('image_path') or r.get('image')
    if not img_path:
        continue
    p = Path(img_path)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.exists():
        # skip missing
        img_data = None
        mime = None
    else:
        b = p.read_bytes()
        # guess mime from suffix
        suff = p.suffix.lower()
        if suff in ('.jpg','.jpeg'):
            mime = 'image/jpeg'
        elif suff in ('.png',):
            mime = 'image/png'
        elif suff in ('.webp',):
            mime = 'image/webp'
        elif suff in ('.gif',):
            mime = 'image/gif'
        else:
            mime = 'application/octet-stream'
        b64 = base64.b64encode(b).decode('ascii')
        img_data = f'data:{mime};base64,{b64}'

    expected = r.get('expected_depth_cm')
    evald = r.get('eval_depth_cm') if r.get('eval_depth_cm') is not None else r.get('depth_cm')
    pred = r.get('predicted_flood')
    expf = r.get('expected_flood')
    sev = r.get('severity_level')
    wconf = r.get('water_confidence')
    wcov = r.get('water_coverage_pct')
    notes = r.get('notes') or ''

    card = ['<div class="card">']
    if img_data:
        card.append(f'<img src="{img_data}" alt="{p.name}">')
    else:
        card.append(f'<div style="height:200px;background:#eee;display:flex;align-items:center;justify-content:center;color:#666">Missing: {p}</div>')
    card.append('<div class="meta">')
    card.append(f'<span class="k">image:</span> {p.name}<br/>')
    card.append(f'<span class="k">path:</span> <code>{p}</code><br/>')
    card.append(f'<span class="k">expected_depth_cm:</span> {expected} &nbsp;&nbsp; <span class="k">eval_depth_cm:</span> {evald} &nbsp;&nbsp; <span class="k">abs_err:</span> {abs(to_float(evald) - to_float(expected)) if to_float(evald) is not None and to_float(expected) is not None else "n/a"}<br/>')
    card.append(f'<span class="k">predicted_flood:</span> {pred} &nbsp;&nbsp; <span class="k">expected_flood:</span> {expf} &nbsp;&nbsp; <span class="k">severity:</span> {sev}<br/>')
    card.append(f'<span class="k">water_confidence:</span> {wconf} &nbsp;&nbsp; <span class="k">water_coverage_pct:</span> {wcov}<br/>')
    card.append(f'<span class="k">notes:</span> {notes}')
    card.append('</div></div>')
    html.append('\n'.join(card))

html.append('</div>')
html.append('<p style="font-size:12px;color:#333">Note: Images are embedded inline. File generated from {}</p>'.format(report_path))
html.append('</body></html>')

out_path.write_text('\n'.join(html), encoding='utf-8')
print('Wrote', out_path.resolve())
print('Open the HTML in your browser to inspect inline images.')
