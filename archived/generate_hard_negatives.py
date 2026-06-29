import argparse
from pathlib import Path

import cv2
import numpy as np


def _write_image(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _make_barren_variant(seed):
    rng = np.random.default_rng(seed)
    img = np.full((448, 448, 3), (95, 105, 115), dtype=np.uint8)
    noise = rng.normal(0, 9, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for _ in range(4):
        y = int(rng.integers(40, 408))
        thickness = int(rng.integers(1, 3))
        color = int(rng.integers(60, 90))
        cv2.line(img, (0, y), (447, y), (color, color, color), thickness)
    return img


def _make_reflection_variant(seed):
    rng = np.random.default_rng(seed)
    img = np.full((448, 448, 3), (70, 80, 90), dtype=np.uint8)
    for _ in range(10):
        x1, y1 = int(rng.integers(0, 448)), int(rng.integers(0, 448))
        x2, y2 = int(rng.integers(0, 448)), int(rng.integers(0, 448))
        color = int(rng.integers(120, 190))
        cv2.line(img, (x1, y1), (x2, y2), (color, color, color), 1)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    return cv2.addWeighted(img, 0.7, blur, 0.3, 0)


def _make_rain_variant(seed):
    rng = np.random.default_rng(seed)
    img = np.full((448, 448, 3), (55, 65, 75), dtype=np.uint8)
    for _ in range(250):
        x = int(rng.integers(0, 448))
        y = int(rng.integers(0, 448))
        length = int(rng.integers(8, 20))
        cv2.line(img, (x, y), (min(447, x + 2), min(447, y + length)), (170, 170, 170), 1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def generate(output_dir, barren_count=25, reflection_count=10, rain_count=10):
    output_dir = Path(output_dir)
    for idx in range(1, barren_count + 1):
        image = _make_barren_variant(seed=idx)
        _write_image(output_dir / f"hardneg_barren_{idx:03d}.jpg", image)
    for idx in range(1, reflection_count + 1):
        image = _make_reflection_variant(seed=1000 + idx)
        _write_image(output_dir / f"hardneg_reflection_{idx:03d}.jpg", image)
    for idx in range(1, rain_count + 1):
        image = _make_rain_variant(seed=2000 + idx)
        _write_image(output_dir / f"hardneg_rain_{idx:03d}.jpg", image)
    print(
        {
            "output_dir": str(output_dir),
            "barren": barren_count,
            "reflection": reflection_count,
            "rain": rain_count,
            "total": barren_count + reflection_count + rain_count,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic hard-negative images for calibration.")
    parser.add_argument("--output-dir", default="test_images/hard_negatives", help="Destination folder")
    parser.add_argument("--barren-count", type=int, default=25)
    parser.add_argument("--reflection-count", type=int, default=10)
    parser.add_argument("--rain-count", type=int, default=10)
    args = parser.parse_args()
    generate(
        output_dir=args.output_dir,
        barren_count=args.barren_count,
        reflection_count=args.reflection_count,
        rain_count=args.rain_count,
    )


if __name__ == "__main__":
    main()
