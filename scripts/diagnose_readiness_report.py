#!/usr/bin/env python3
"""Diagnostics: print top-N depth absolute errors and contradictions from a readiness JSON report."""
import json
import sys
from pathlib import Path
import argparse

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

parser = argparse.ArgumentParser()
parser.add_argument('report', help='Path to JSON report')
parser.add_argument('--top', type=int, default=20, help='Top N errors to show')
args = parser.parse_args()
report_path = Path(args.report)
if not report_path.exists():
    print('Report not found:', report_path)
    sys.exit(2)

data = json.loads(report_path.read_text(encoding='utf-8'))
rows = data.get('rows', [])

errors = []
for r in rows:
    exp = to_float(r.get('expected_depth_cm'))
    ev = to_float(r.get('eval_depth_cm') or r.get('depth_cm'))
    if exp is None or ev is None:
        continue
    ae = abs(ev - exp)
    errors.append((ae, r))

errors.sort(key=lambda x: x[0], reverse=True)

print('\nReport:', report_path)
print('Total rows in report:', len(rows))
print('Rows with depth labels (used for MAE):', len(errors))
print('\nTop {} depth absolute errors:'.format(args.top))
print('{:<4} {:<30} {:<8} {:<8} {:<8} {:<6} {:<6} {:<8} {:<6} {:<6}'.format('#','image','expected','eval','abs_err','pred','expf','severity','w_conf','w_cov'))
for i,(ae,r) in enumerate(errors[:args.top], start=1):
    img = r.get('image') or r.get('image_path')
    expected = r.get('expected_depth_cm')
    evald = r.get('eval_depth_cm') if r.get('eval_depth_cm') is not None else r.get('depth_cm')
    pred = r.get('predicted_flood')
    expf = r.get('expected_flood')
    sev = r.get('severity_level')
    wconf = r.get('water_confidence')
    wcov = r.get('water_coverage_pct')
    print('{:<4} {:<30} {:<8} {:<8} {:<8.2f} {:<6} {:<6} {:<8} {:<6} {:<6}'.format(i, img[:30], str(expected)[:8], str(evald)[:8], ae, str(pred), str(expf), str(sev), str(wconf), str(wcov)))

# contradictions
contradictions = [r for r in rows if r.get('contradiction') in (True, 'True', 'true', 1, '1')]
print('\nContradictions found:', len(contradictions))
if contradictions:
    print('{:<4} {:<30} {:<6} {:<6} {:<8} {:<8} {:<8} {:<8}'.format('#','image','pred','expf','depth','exp_depth','severity','notes'))
    for i,r in enumerate(contradictions, start=1):
        img = r.get('image') or r.get('image_path')
        pred = r.get('predicted_flood')
        expf = r.get('expected_flood')
        depth = r.get('depth_cm') or r.get('eval_depth_cm')
        expd = r.get('expected_depth_cm')
        sev = r.get('severity_level')
        notes = (r.get('notes') or '')[:40]
        print('{:<4} {:<30} {:<6} {:<6} {:<8} {:<8} {:<8} {:<8}'.format(i, img[:30], str(pred), str(expf), str(depth)[:8], str(expd)[:8], str(sev), notes))

# also print a small sample of worst errors with full paths (for easy opening)
print('\nFull paths for top errors:')
for ae,r in errors[:args.top]:
    print(r.get('image_path'))

print('\nDone.')
