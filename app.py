from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
import os
from pathlib import Path
import json
import cv2
import numpy as np
# Load configuration helper (reads utils/config.cfg and can export env vars)
from utils.config import load_config, get_config

# Make modules/ importable both as package and as top-level modules (matches main.py behavior)
import sys
sys.path.insert(0, str(Path(__file__).parent / "modules"))

# Don't initialize S3 or heavy models at import-time; do it lazily in request handlers

app = Flask(__name__)

# Load config file (if present) and export env vars when missing
try:
    load_config(export_env=True)
except Exception:
    # Silence missing config — environment variables may be used instead
    pass

# Read merged config (prefers env vars when set)
_CFG = get_config()

app.secret_key = os.getenv('FLASK_SECRET', _CFG.get('app', {}).get('flask_secret', 'change-me'))

# Configuration (prefer env vars, then config file defaults)
DEFAULT_S3_BUCKET = os.getenv('S3_BUCKET', _CFG.get('aws', {}).get('s3_bucket', 'application-testing-tsg'))
S3_PREFIX = os.getenv('S3_PREFIX', _CFG.get('aws', {}).get('s3_prefix', 'input/test_images/'))
LOCAL_OUTPUT_DIR = Path(__file__).parent / 'static' / 'output_images'
LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_s3_handler():
    """Create and return an S3Handler instance using configured bucket."""
    from modules.s3_handler import S3Handler
    bucket = os.getenv('S3_BUCKET', DEFAULT_S3_BUCKET)
    return S3Handler(bucket_name=bucket)


def save_overlay_image(image_bgr, mask, out_local_path: Path):
    """Save a semi-transparent overlay of mask on image to out_local_path."""
    img = image_bgr.copy()
    if mask is None:
        cv2.imwrite(str(out_local_path), img)
        return

    # Ensure mask boolean
    if mask.dtype != np.bool_:
        mask_bool = mask.astype(bool)
    else:
        mask_bool = mask

    colored = np.zeros_like(img)
    colored[mask_bool] = (0, 0, 255)  # red overlay
    alpha = 0.4
    overlay = cv2.addWeighted(img, 1.0, colored, alpha, 0)
    cv2.imwrite(str(out_local_path), overlay)


def sanitize_for_json(obj, key_name=None):
    """Recursively convert objects to JSON-serializable types.

    - Converts numpy types to native Python types
    - Converts Path to str
    - Replaces large numpy arrays (like masks) with a brief descriptor to avoid huge JSON
    - Drops raw image data by key name 'water_mask'
    """
    # Avoid including large mask arrays in JSON
    if key_name == 'water_mask':
        return None

    # Primitives
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Numpy scalar types
    try:
        import numpy as _np
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, (_np.bool_,)):
            return bool(obj)
        if isinstance(obj, _np.ndarray):
            # If array is huge, return a small descriptor
            if obj.size > 100000:
                return f"<ndarray shape={obj.shape} size={obj.size}>"
            return obj.tolist()
    except Exception:
        pass

    # Path
    if isinstance(obj, Path):
        return str(obj)

    # Dict
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out[k] = sanitize_for_json(v, key_name=k)
        return out

    # List / tuple
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]

    # Fallback to string
    try:
        return str(obj)
    except Exception:
        return None


@app.route('/')
def index():
    # List images from S3 under prefix
    try:
        s3 = get_s3_handler()
        keys = s3.list_images_in_s3(prefix=S3_PREFIX)
    except Exception as e:
        keys = []
        # don't flood the UI with S3 errors; show local images instead
        flash(f"S3 not available or credentials missing: {e}")

    # Also list local images (fallback) from repository test_images/
    local_dir = Path(__file__).parent / 'test_images'
    local_images = []
    if local_dir.exists():
        for p in sorted(local_dir.iterdir()):
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # value format: local:<relative_path>
                rel = str(Path('test_images') / p.name)
                local_images.append(rel)

    # Normalize keys for display
    s3_images = [k for k in keys if k.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return render_template('index.html', images=s3_images, local_images=local_images, bucket=os.getenv('S3_BUCKET', DEFAULT_S3_BUCKET))


@app.route('/analyze', methods=['POST'])
def analyze():
    s3_key = request.form.get('s3_key')
    mode = request.form.get('mode', 'image')

    if not s3_key:

        flash('Please select an image')
        return redirect(url_for('index'))

    # Initialize handlers lazily
    s3 = None
    image = None

    try:
        # Determine if the selected key is local or s3
        if s3_key.startswith('local:'):
            # local:<relative_path_from_repo_root>
            local_rel = s3_key.split(':', 1)[1]
            local_path = Path(__file__).parent / local_rel
            image = cv2.imread(str(local_path))
            if image is None:
                raise FileNotFoundError(f"Local image not found: {local_path}")
            source_is_local = True
        else:
            # attempt S3
            s3 = get_s3_handler()
            image = s3.read_image_from_s3(s3_key)
            source_is_local = False

        result = None
        annotated_local = None
        annotated_s3_key = None

        if mode == 'image':
            from modules.flood_analyzer import FloodAnalyzer
            analyzer = FloodAnalyzer()
            result = analyzer.analyze_bgr(image, s3_key)

            # Save overlay locally and upload to S3
            img_name = Path(s3_key).name
            local_out = LOCAL_OUTPUT_DIR / f"overlay_{img_name}"
            save_overlay_image(image, result.get('water_mask'), local_out)
            annotated_local = str(local_out)
            annotated_s3_key = None
            # If S3 available and source was S3, upload annotated and JSON
            if (s3 is not None) and (not source_is_local):
                try:
                    annotated_s3_key = f"output/annotated/{img_name}"
                    s3.write_image_to_s3(cv2.imread(annotated_local), annotated_s3_key)

                    # Save JSON result to S3 under output/results/
                    json_key = f"output/results/{img_name}.json"
                    sanitized = sanitize_for_json(result)
                    json_string = json.dumps(sanitized)
                    s3.write_csv_to_s3(json_string, json_key)
                except Exception:
                    # ignore upload errors; local files are still available
                    annotated_s3_key = None

        elif mode == 'object':
            from modules.object_detection import ObjectDetector
            detector = ObjectDetector()
            detections = detector.detect_objects(image)

            # Draw annotated image
            annotated = detector.draw_detections(image, detections)
            img_name = Path(s3_key).name
            local_out = LOCAL_OUTPUT_DIR / f"objects_{img_name}"
            cv2.imwrite(str(local_out), annotated)
            annotated_local = str(local_out)
            annotated_s3_key = None
            if (s3 is not None) and (not source_is_local):
                try:
                    annotated_s3_key = f"output/objects/{img_name}"
                    s3.write_image_to_s3(annotated, annotated_s3_key)
                except Exception:
                    annotated_s3_key = None

            result = {
                's3_key': s3_key,
                'detections': detections,
            }

            # Save JSON result
            # upload JSON if possible
            if (s3 is not None) and (not source_is_local):
                try:
                    json_key = f"output/results/objects_{img_name}.json"
                    sanitized = sanitize_for_json(result)
                    json_string = json.dumps(sanitized)
                    s3.write_csv_to_s3(json_string, json_key)
                except Exception:
                    pass

        else:
            flash('Unknown mode')
            return redirect(url_for('index'))

        # Prepare display URLs
        # Make annotated_local path relative to static
        rel_path = None
        if annotated_local:
            rel_path = os.path.relpath(annotated_local, Path(__file__).parent / 'static')

        # Sanitize result for template JSON rendering (Jinja tojson uses json.dumps)
        display_result = sanitize_for_json(result) if result is not None else None
        return render_template('result.html', result=display_result, annotated_url='/' + str(Path('static') / rel_path) if rel_path else None, s3_bucket=(s3.bucket_name if s3 else DEFAULT_S3_BUCKET), s3_key=s3_key, annotated_s3_key=annotated_s3_key)

    except Exception as e:
        flash(f"Error during analysis: {e}")
        return redirect(url_for('index'))


if __name__ == '__main__':
    # Run development server
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

