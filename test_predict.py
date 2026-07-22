"""
Quick test script — run from repo root:
  py test_predict.py path/to/flood_image.jpg

Tests the running serve.py server at localhost:8000
"""

import sys, base64, json, urllib.request, pathlib

SERVER = "http://localhost:8000/predict"

def test_image(img_path: str):
    path = pathlib.Path(img_path)
    if not path.exists():
        print(f"ERROR: File not found: {img_path}")
        sys.exit(1)

    print(f"Testing: {path.name} ({path.stat().st_size / 1024:.1f} KB)")

    b64 = base64.b64encode(path.read_bytes()).decode()

    payload = json.dumps({
        "images": [{
            "image_b64":  b64,
            "camera_id":  "test-cam-01",
            "latitude":   12.9716,   # Bengaluru
            "longitude":  77.5946,
            "source":     "serve",
        }]
    }).encode()

    req = urllib.request.Request(
        SERVER,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            print(json.dumps(result, indent=2))
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"HTTP {e.code}: {body}")
    except ConnectionRefusedError:
        print("ERROR: Server not running. Start it with:  py serve.py")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: use first flood image found in archived/
        import glob
        imgs = glob.glob("archived/**/*.jpg", recursive=True)
        if not imgs:
            print("Usage: py test_predict.py path/to/image.jpg")
            sys.exit(1)
        test_image(imgs[0])
    else:
        test_image(sys.argv[1])
