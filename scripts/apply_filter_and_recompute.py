#!/usr/bin/env python3
"""Apply quick heuristic filters to an existing readiness JSON report and recompute metrics.
Generates two filtered reports:
 - *_filtered_h1.json: keep predicted flood only when water_confidence>=0.6 AND water_coverage_pct>=50
 - *_filtered_h2.json: same as h1 AND no_reference_warning is False

Usage: py -3 scripts\apply_filter_and_recompute.py reports\model_readiness_YYYY.json
"""
import json
import sys
from pathlib import Path
from copy import deepcopy
from statistics import mean

if len(sys.argv) < 2:
    print('Usage: apply_filter_and_recompute.py <report.json>')
    sys.exit(2)

report_path = Path(sys.argv[1])
if not report_path.exists():
    print('Report not found:', report_path)
    sys.exit(2)

data = json.loads(report_path.read_text(encoding='utf-8'))
rows = data.get('rows', [])

BANDS = [(0,20,'0_20'), (20,50,'20_50'), (50,80,'50_80'), (80,99999,'80_plus')]

def to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except:
        try:
            return float(str(v).strip())
        except:
            return None

def boolish(v):
    if v in (True, False):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ('1','true','t','yes','y')

def recompute(rows_modified):
    # classification counts
    tp = tn = fp = fn = 0
    barren_total = 0
    barren_fp = 0
    depth_errors = []
    per_band = {k: [] for (_,_,k) in BANDS}
    contradictions = 0
    latencies = []
    for r in rows_modified:
        pred = 1 if str(r.get('predicted_flood','0')).strip() in ('1','True','true','true') else 0
        expf = 1 if str(r.get('expected_flood','0')).strip() in ('1','True','true','true') else 0
        # classification
        if pred==1 and expf==1:
            tp += 1
        elif pred==1 and expf==0:
            fp += 1
        elif pred==0 and expf==0:
            tn += 1
        elif pred==0 and expf==1:
            fn += 1
        # barren
        scene = (r.get('scene_type') or '').lower()
        if expf==0:
            barren_total += 1
            if pred==1:
                barren_fp += 1
        # contradictions: predicted non-flood but expected flood
        if pred==0 and expf==1:
            contradictions += 1
        # depth errors: only where expected_depth exists AND pred==1
        expd = to_float(r.get('expected_depth_cm'))
        evald = to_float(r.get('eval_depth_cm') if r.get('eval_depth_cm') is not None else r.get('depth_cm'))
        if expd is not None and evald is not None and pred==1:
            ae = abs(evald - expd)
            depth_errors.append(ae)
            # band
            for lo,hi,k in BANDS:
                if lo <= expd < hi:
                    per_band[k].append(ae)
                    break
        # latency
        lat = to_float(r.get('elapsed_sec'))
        if lat is not None:
            latencies.append(lat)
    precision = tp/(tp+fp) if (tp+fp)>0 else None
    recall = tp/(tp+fn) if (tp+fn)>0 else None
    f1 = (2*precision*recall/(precision+recall)) if precision and recall and (precision+recall)>0 else None
    depth_mae = mean(depth_errors) if depth_errors else None
    per_band_mae = {k: (mean(v) if v else None) for k,v in per_band.items()}
    p95 = None
    if latencies:
        lat_sorted = sorted(latencies)
        idx = max(0, int(0.95*len(lat_sorted))-1)
        p95 = lat_sorted[idx]
    barren_fp_rate = barren_fp / barren_total if barren_total>0 else None
    summary = {
        'tp': tp,'tn':tn,'fp':fp,'fn':fn,'precision':precision,'recall':recall,'f1':f1,
        'barren_total':barren_total,'barren_fp':barren_fp,'barren_fp_rate':barren_fp_rate,
        'depth_mae_cm': depth_mae,'depth_mae_by_band': {k:{'count':len(per_band[k]), 'mae_cm': per_band_mae[k]} for (_,_,k) in BANDS},
        'contradiction_count': contradictions,'p95_latency_sec': p95
    }
    return summary

# define filters

def apply_filter(rows, mode='h1'):
    rows2 = deepcopy(rows)
    for r in rows2:
        pred = 1 if str(r.get('predicted_flood','0')).strip() in ('1','True','true','true') else 0
        if pred!=1:
            continue
        wconf = to_float(r.get('water_confidence')) or 0.0
        wcov = to_float(r.get('water_coverage_pct')) or 0.0
        no_ref = boolish(r.get('no_reference_warning'))
        # h1: require both wconf>=0.6 and wcov>=50
        h1_keep = (wconf >= 0.6) and (wcov >= 50.0)
        # h2: h1 AND not no_reference_warning
        h2_keep = h1_keep and (not no_ref)
        if mode=='h1':
            keep = h1_keep
        else:
            keep = h2_keep
        if not keep:
            # suppress prediction
            r['predicted_flood'] = 0
            # remove predicted depth
            r['eval_depth_cm'] = None
            r['depth_cm'] = None
            r['severity_level'] = None
            r['confidence'] = None
    return rows2

# apply filters and recompute
rows_h1 = apply_filter(rows, 'h1')
rows_h2 = apply_filter(rows, 'h2')

summary_h1 = recompute(rows_h1)
summary_h2 = recompute(rows_h2)
summary_orig = recompute(rows)

base = report_path.stem
out1 = report_path.with_name(base + '_filtered_h1.json')
out2 = report_path.with_name(base + '_filtered_h2.json')

rep1 = {'filtered_mode':'h1','orig_report':str(report_path),'summary':summary_h1,'rows':rows_h1}
rep2 = {'filtered_mode':'h2','orig_report':str(report_path),'summary':summary_h2,'rows':rows_h2}

out1.write_text(json.dumps(rep1, indent=2), encoding='utf-8')
out2.write_text(json.dumps(rep2, indent=2), encoding='utf-8')

print('Original summary:')
print(json.dumps(summary_orig, indent=2))
print('\nAfter filter h1 (wconf>=0.6 and wcov>=50):')
print(json.dumps(summary_h1, indent=2))
print('\nAfter filter h2 (h1 AND no_reference_warning==False):')
print(json.dumps(summary_h2, indent=2))
print('\nWrote:', out1)
print('Wrote:', out2)
print('\nNote: These are simulated filters applied to the last run report only. For faithful CI metrics rerun, run evaluate_model_readiness.py with a pipeline-level filter or update the pipeline.' )
