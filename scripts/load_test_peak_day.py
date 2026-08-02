from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import requests


def _collect_images(image_dir: Path, max_images: int) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [p for p in sorted(image_dir.rglob("*")) if p.is_file() and p.suffix.lower() in exts]
    if max_images > 0:
        images = images[:max_images]
    return images


def _post_predict(url: str, image_path: Path, timeout_seconds: float) -> Tuple[float, int]:
    with image_path.open("rb") as handle:
        files = {"image": (image_path.name, handle, "image/jpeg")}
        data = {"camera_id": "peak-load-test", "latitude": "12.9716", "longitude": "77.5946"}
        start = time.perf_counter()
        resp = requests.post(url, files=files, data=data, timeout=timeout_seconds)
        elapsed = (time.perf_counter() - start) * 1000.0
        return elapsed, int(resp.status_code)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((p / 100.0) * len(ordered))) - 1))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Peak-day /predict load test rehearsal.")
    parser.add_argument("--target-url", required=True, help="Target /predict URL")
    parser.add_argument("--image-dir", default="test_images", help="Directory with test images")
    parser.add_argument("--max-images", type=int, default=200, help="Total request count")
    parser.add_argument("--concurrency", type=int, default=20, help="Parallel request workers")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-p95-ms", type=float, default=8000.0)
    parser.add_argument("--min-success-rate", type=float, default=0.95)
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    if not image_dir.exists():
        raise SystemExit(f"image-dir not found: {image_dir}")
    images = _collect_images(image_dir, args.max_images)
    if not images:
        raise SystemExit("No images found for load test.")

    latencies: List[float] = []
    statuses: List[int] = []
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(_post_predict, args.target_url, img, args.timeout_seconds) for img in images]
        for fut in as_completed(futures):
            elapsed_ms, status_code = fut.result()
            latencies.append(elapsed_ms)
            statuses.append(status_code)

    success = sum(1 for code in statuses if 200 <= code < 300)
    success_rate = success / len(statuses)
    p95_ms = _percentile(latencies, 95.0)
    avg_ms = statistics.mean(latencies)

    print(
        {
            "requests": len(statuses),
            "success_rate": round(success_rate, 4),
            "avg_latency_ms": round(avg_ms, 3),
            "p95_latency_ms": round(p95_ms, 3),
            "max_p95_ms": args.max_p95_ms,
            "min_success_rate": args.min_success_rate,
        }
    )

    if success_rate < args.min_success_rate:
        raise SystemExit(f"Load test failed: success_rate={success_rate:.4f} < {args.min_success_rate:.4f}")
    if p95_ms > args.max_p95_ms:
        raise SystemExit(f"Load test failed: p95_latency_ms={p95_ms:.3f} > {args.max_p95_ms:.3f}")


if __name__ == "__main__":
    main()
