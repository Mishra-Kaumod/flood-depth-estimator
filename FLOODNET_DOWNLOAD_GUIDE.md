# FloodNet Download Script

## Overview

`download_floodnet.py` downloads the FloodNet dataset from Hugging Face, verifies image-mask pairs, and generates an inventory report.

## Features

✓ Downloads FloodNet from `torchgeo/floodnet` on Hugging Face  
✓ Saves raw images to `datasets/floodnet/raw/images/`  
✓ Saves raw masks to `datasets/floodnet/raw/masks/`  
✓ Verifies image-mask pair compatibility  
✓ Generates `datasets/floodnet/raw_inventory.csv` with detailed metadata  

## Requirements

```bash
pip install datasets pillow numpy
```

The script requires ~13 GB of free disk space to download the full FloodNet dataset (1,400 images with masks).

## Usage

### Basic Download and Verify

```bash
python download_floodnet.py
```

This will:
1. Create output directories (`datasets/floodnet/raw/images/` and `datasets/floodnet/raw/masks/`)
2. Download FloodNet from Hugging Face
3. Save raw image and mask files
4. Verify all pairs for size and format compatibility
5. Generate inventory CSV report

### Verify Existing Files Only

```bash
python download_floodnet.py --verify-only
```

Use this to re-verify or regenerate the inventory for existing files without re-downloading.

### Custom Output Directory

```bash
python download_floodnet.py --output-base /path/to/custom/dir
```

## Output

### Directory Structure

```
datasets/floodnet/
├── raw/
│   ├── images/              # PNG image files (floodnet_0000.png, floodnet_0001.png, ...)
│   ├── masks/               # PNG mask files (floodnet_0000_mask.png, ...)
│   └── raw_inventory.csv    # Inventory metadata and verification report
```

### Inventory CSV Columns

| Column | Description |
|--------|-------------|
| `image_file` | Original image filename |
| `mask_file` | Original mask filename |
| `image_path` | Full path to image |
| `mask_path` | Full path to mask |
| `image_shape` | Image dimensions (width×height) |
| `mask_shape` | Mask dimensions (width×height) |
| `image_dtype` | Image data type (uint8, etc.) |
| `mask_dtype` | Mask data type |
| `image_channels` | Image channels (ndim) |
| `mask_channels` | Mask channels (ndim) |
| `size_match` | Boolean: dimensions match |
| `status` | VALID, SIZE_MISMATCH, MISSING_MASK, ERROR |
| `notes` | Additional details/error messages |

### Sample Output

```
INFO:root:Directories ready: datasets/floodnet/raw
INFO:root:Loading FloodNet dataset from Hugging Face (torchgeo/floodnet)...
INFO:root:Loaded dataset with 1400 samples
INFO:root:Processing and saving samples...
INFO:root:Processed 100 samples...
INFO:root:Processed 200 samples...
...
INFO:root:Verifying image-mask pairs...
INFO:root:Found 1400 image files
INFO:root:Generating inventory report: datasets/floodnet/raw_inventory.csv
INFO:root:Inventory saved to datasets/floodnet/raw_inventory.csv

============================================================
FLOODNET DOWNLOAD SUMMARY
============================================================
Output directory:    datasets/floodnet/raw
Total downloaded:    1400
Valid pairs:         1400
Invalid pairs:       0
Size mismatches:     0
Processing errors:   0
Inventory report:    datasets/floodnet/raw_inventory.csv
============================================================
```

## Example: Analyzing the Inventory

Once the download completes, analyze the inventory:

```python
import pandas as pd

# Load inventory
inventory = pd.read_csv("datasets/floodnet/raw_inventory.csv")

# Check status summary
print(inventory['status'].value_counts())

# Filter valid pairs
valid = inventory[inventory['status'] == 'VALID']
print(f"Valid pairs: {len(valid)}")

# Check image dimensions
print(inventory['image_shape'].unique())
```

## Disk Space

- **Download size**: ~13 GB (compressed dataset)
- **Extracted size**: ~10–12 GB (decompressed images + masks)
- **Inventory file**: ~1 MB (CSV)
- **Total needed**: ~15–16 GB free

## Troubleshooting

**Issue**: "Not enough free disk space"
- Free up at least 16 GB on your disk
- Or use an external/mounted storage with more space
- Use `--output-base /mnt/large-disk/floodnet` to specify alternate location

**Issue**: Network timeout during download
- Increase your internet connection timeout
- Try running in a more stable network environment
- Rerun the script to resume (partially cached files will be reused)

**Issue**: "MISSING_MASK" or "SIZE_MISMATCH" pairs
- These are flagged in the inventory CSV
- Filter by `status != 'VALID'` to see problematic files
- Remove mismatched pairs before training

## Next Steps

After successful download and verification:

1. **Inspect masks**: Check that masks are binary or multi-class as expected
2. **Preprocessing**: Use masks as-is or convert to binary water/non-water labels
3. **Training**: Feed `datasets/floodnet/raw/` into `train_deeplab.py`

## License

FloodNet dataset uses the `cdla-permissive-1.0` license. Always cite the original paper:

```
@article{rahnemoonfar2020floodnet,
  title={FloodNet: A high resolution aerial imagery dataset for post flood scene understanding},
  author={Rahnemoonfar, Maryam and Chowdhury, Tashnim},
  journal={arXiv preprint arXiv:2012.02951},
  year={2020}
}
```

## See Also

- [FLOOD_DATASET_SOURCES.md](FLOOD_DATASET_SOURCES.md) — Other publicly available flood datasets
- [DEEPLAB_TRAINING_PLAN.md](DEEPLAB_TRAINING_PLAN.md) — Training plan for DeepLabV3 models
- [train_deeplab.py](train_deeplab.py) — Training script using downloaded data
