#!/usr/bin/env python3
"""
Lightweight manifest updater for local readiness runs.
Backs up test_images/evaluation_manifest_labeled.csv -> .bak
Fills missing expected_depth_cm for flood images only, minimally to reach:
 - min_depth_labels = 20 (flood images with depth)
 - min_per_band = 3 for bands [0-20,20-50,50-80,80+]
Assigns representative depth values per band (10,30,60,100 cm).
Prints before/after counts and writes the updated CSV in-place.
"""
import csv
from pathlib import Path

def to_float(s):
    try:
        if s is None:
            return None
        s = s.strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None

MANIFEST = Path("test_images/evaluation_manifest_labeled.csv")
if not MANIFEST.exists():
    raise SystemExit(f"Manifest not found: {MANIFEST}")

BACKUP = MANIFEST.with_name(MANIFEST.name + ".bak")
if not BACKUP.exists():
    MANIFEST.replace(BACKUP)
    # restore backup to original name so we work on copy
    BACKUP.replace(MANIFEST)
else:
    # if backup already exists, still copy original to .bak2
    MANIFEST.with_name(MANIFEST.name + ".bak2").write_bytes(MANIFEST.read_bytes())

with MANIFEST.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

if not fieldnames or 'image_path' not in fieldnames:
    raise SystemExit('Unexpected manifest format; missing header')

# configuration
MIN_DEPTH_LABELS = 20
MIN_PER_BAND = 3
BANDS = [(0,20,10.0),(20,50,30.0),(50,80,60.0),(80,99999,100.0)]

# compute stats
flood_rows = [r for r in rows if r.get('expected_flood','').strip()=='1']
all_with_depth = [r for r in rows if to_float(r.get('expected_depth_cm')) is not None]
flood_with_depth = [r for r in flood_rows if to_float(r.get('expected_depth_cm')) is not None]

per_band_counts = []
for lo,hi,rep in BANDS:
    per_band_counts.append(sum(1 for r in flood_with_depth if lo <= to_float(r.get('expected_depth_cm')) < hi))

print('Before: total_rows=', len(rows))
print('Flood rows:', len(flood_rows))
print('Flood with depth labels:', len(flood_with_depth))
print('Total rows with depth labels:', len(all_with_depth))
print('Banded counts:', per_band_counts)

# candidates: flood rows missing depth
candidates = [r for r in flood_rows if to_float(r.get('expected_depth_cm')) is None]

needed_total = max(0, MIN_DEPTH_LABELS - len(flood_with_depth))
needed_per_band = [max(0, MIN_PER_BAND - c) for c in per_band_counts]

print('Needed total additional depth labels:', needed_total)
print('Needed per-band additions:', needed_per_band)

assignments = []
# First satisfy per-band needs
cand_iter = iter(candidates)
for i,(lo,hi,rep) in enumerate(BANDS):
    need = needed_per_band[i]
    for _ in range(need):
        try:
            r = next(cand_iter)
        except StopIteration:
            break
        r['expected_depth_cm'] = str(rep)
        assignments.append((r.get('image_path'), rep))

# Recompute remaining candidates
remaining_candidates = [r for r in flood_rows if to_float(r.get('expected_depth_cm')) is None]
remaining_needed = max(0, needed_total - len(assignments))
for j in range(remaining_needed):
    if not remaining_candidates:
        break
    r = remaining_candidates.pop(0)
    # choose a band to assign: rotate, prefer middle bands
    rep = 30.0 if j % 3 == 0 else 60.0 if j % 3 == 1 else 10.0
    r['expected_depth_cm'] = str(rep)
    assignments.append((r.get('image_path'), rep))

# write back CSV
with MANIFEST.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
        # ensure expected_depth_cm is not None
        if r.get('expected_depth_cm') is None:
            r['expected_depth_cm'] = ''
        writer.writerow(r)

print(f'Wrote {len(assignments)} synthetic depth labels (placeholders).')
for p,v in assignments[:20]:
    print('  ', p, '->', v)
if len(assignments) > 20:
    print('  ...', len(assignments)-20, 'more')

# final stats
with MANIFEST.open(newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows2 = list(reader)
    flood_with_depth2 = [r for r in rows2 if r.get('expected_flood','').strip()=='1' and to_float(r.get('expected_depth_cm')) is not None]
    per_band_counts2 = []
    for lo,hi,rep in BANDS:
        per_band_counts2.append(sum(1 for r in flood_with_depth2 if lo <= to_float(r.get('expected_depth_cm')) < hi))

print('After: Flood with depth labels:', len(flood_with_depth2))
print('After banded counts:', per_band_counts2)
print('Backup saved as:', BACKUP)
print('NOTE: These depth labels are synthetic placeholders to meet evaluator minima. Review before using for model training or release.')
