"""
Auto-Download Flood Images from Public Sources
===============================================
Supplements your manual uploads with pre-labeled datasets from:
- Kaggle Flood Datasets
- Google URLs
- GitHub repos
- Public archives

Run this in Colab BEFORE uploading your own images
"""

import os
import shutil
import urllib.request
import urllib.error
from pathlib import Path
import json
from tqdm import tqdm

def download_from_urls(urls, output_dir='flood_images_raw', max_workers=4):
    """Download images from a list of URLs"""
    os.makedirs(output_dir, exist_ok=True)
    downloaded = 0
    failed = 0
    
    print(f"📥 Downloading from {len(urls)} URLs...")
    for i, url in enumerate(tqdm(urls, desc="Downloading")):
        try:
            filename = f"web_{i:04d}.jpg"
            urllib.request.urlretrieve(url, f'{output_dir}/{filename}')
            downloaded += 1
        except Exception as e:
            failed += 1
    
    print(f"✅ Downloaded: {downloaded}, Failed: {failed}")
    return downloaded


def download_kaggle_dataset(dataset_id, output_dir='flood_images_raw'):
    """Download from Kaggle (requires API key)"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📥 Downloading Kaggle dataset: {dataset_id}")
    os.system(f"kaggle datasets download -d {dataset_id} -p {output_dir} --unzip")
    
    count = len([f for f in os.listdir(output_dir) 
                 if f.lower().endswith(('.jpg','.jpeg','.png'))])
    print(f"✅ Downloaded {count} images")
    return count


def filter_images_by_size(input_dir, output_dir, min_size=(200, 200)):
    """Filter images by resolution (discard tiny ones)"""
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    filtered = 0
    
    print(f"\n🔍 Filtering images by size (min {min_size[0]}×{min_size[1]})...")
    for f in tqdm(os.listdir(input_dir)):
        if not f.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        try:
            img = Image.open(f'{input_dir}/{f}')
            if img.size[0] >= min_size[0] and img.size[1] >= min_size[1]:
                shutil.copy2(f'{input_dir}/{f}', f'{output_dir}/{f}')
                filtered += 1
        except:
            pass
    
    print(f"✅ Kept {filtered} good-resolution images")
    return filtered


def convert_to_rgb(input_dir, output_dir):
    """Convert RGBA/grayscale to RGB"""
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    converted = 0
    
    print(f"\n🎨 Converting images to RGB...")
    for f in tqdm(os.listdir(input_dir)):
        if not f.lower().endswith(('.jpg','.jpeg','.png')):
            continue
        try:
            img = Image.open(f'{input_dir}/{f}')
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(f'{output_dir}/{f}', 'JPEG', quality=95)
            converted += 1
        except:
            pass
    
    print(f"✅ Converted {converted} images to RGB")
    return converted


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC FLOOD IMAGE URLs (carefully curated, free-to-use datasets)
# ═════════════════════════════════════════════════════════════════════════════

FLOOD_IMAGE_URLS = [
    # Wikimedia Commons - Flood Category (public domain)
    "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/2011_River_Brahmaputra_Flood.jpg/1024px-2011_River_Brahmaputra_Flood.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Bangkok_flood_November_2011.jpg/1024px-Bangkok_flood_November_2011.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Kosi_River_Flooded_2008.jpg/1024px-Kosi_River_Flooded_2008.jpg",
    
    # NOAA (public domain disaster images)
    "https://www.ncei.noaa.gov/data/",  # requires manual navigation
    
    # More Wikimedia examples
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/2012_Pakistan_floods.jpg/1024px-2012_Pakistan_floods.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Thailand_Flood_2011_Pathum_Thani.jpg/1024px-Thailand_Flood_2011_Pathum_Thani.jpg",
]

KAGGLE_DATASETS = [
    # Real flood detection datasets (free tier available)
    "emmettsalzano/flood-detection-data-from-space",  # Satellite imagery
    "agrawalshray/flood-detection-segmentation-dataset",  # Street-level
    "rounak441/flood-damage-assessments",  # Damage + depth
]


# ═════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION (for Colab)
# ═════════════════════════════════════════════════════════════════════════════

def main():
    """Main workflow: download + filter + prepare"""
    
    print("🌊 FLOOD IMAGE AUTO-DOWNLOADER")
    print("=" * 60)
    
    # Step 1: Download from URLs
    print("\n📥 STEP 1: Downloading from public URLs...")
    if FLOOD_IMAGE_URLS:
        download_from_urls(FLOOD_IMAGE_URLS, 'flood_images_raw')
    
    # Step 2: Try Kaggle (if API available)
    print("\n📥 STEP 2: Attempting Kaggle downloads...")
    try:
        for dataset in KAGGLE_DATASETS[:1]:  # Start with 1 to test
            try:
                download_kaggle_dataset(dataset, 'flood_images_raw')
                break  # Stop after first success
            except:
                continue
    except:
        print("⚠️  Kaggle not configured (need kaggle.json)")
    
    # Step 3: Filter by size
    print("\n🔍 STEP 3: Filtering images...")
    filter_images_by_size('flood_images_raw', 'flood_images_filtered', min_size=(200, 200))
    
    # Step 4: Convert to RGB
    print("\n🎨 STEP 4: Normalizing formats...")
    convert_to_rgb('flood_images_filtered', 'flood_images_ready')
    
    # Step 5: Summary
    final_count = len([f for f in os.listdir('flood_images_ready/') 
                       if f.lower().endswith(('.jpg','.jpeg','.png'))])
    
    print("\n" + "=" * 60)
    print(f"✅ COMPLETE: {final_count} images ready for training")
    print(f"   Location: flood_images_ready/")
    print("\n📝 Next steps:")
    print("   1. Upload YOUR OWN images (via files.upload())")
    print("   2. Merge with auto-downloaded images")
    print("   3. Label with Gemini Pro")
    print("   4. Train!")


if __name__ == "__main__":
    main()
