import argparse
import csv
from pathlib import Path


URBAN_MARKERS = (
    "city",
    "urban",
    "street",
    "road",
    "building",
    "bridge",
    "traffic",
    "highway",
    "junction",
)
RURAL_MARKERS = (
    "rural",
    "village",
    "farm",
    "field",
    "river",
    "forest",
    "tree",
    "mountain",
    "canal",
)


def _to_int(value):
    text = ("" if value is None else str(value)).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _to_float(value):
    text = ("" if value is None else str(value)).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _infer_scene_type(row, expected_flood):
    scene = (row.get("scene_type") or "").strip().lower()
    image_name = Path((row.get("image_path") or "")).name.lower()
    ai_scene = (row.get("ai_suggested_scene_type") or "").strip().lower()

    if scene == "barren" or ai_scene == "barren":
        return "barren"

    merged = f"{image_name} {scene} {ai_scene}"
    if any(marker in merged for marker in RURAL_MARKERS):
        return "rural"
    if any(marker in merged for marker in URBAN_MARKERS):
        return "urban"

    if expected_flood == 0:
        return "barren"
    # Default unresolved flooded scenes to urban for conservative traffic-risk deployment assumptions.
    return "urban" if expected_flood == 1 else "rural"


def _resolve_expected_flood(row):
    expected = _to_int(row.get("expected_flood"))
    if expected in (0, 1):
        return expected
    suggested = _to_int(row.get("ai_suggested_flood"))
    if suggested in (0, 1):
        return suggested
    return None


def _resolve_expected_depth(row, expected_flood):
    depth = _to_float(row.get("expected_depth_cm"))
    if depth is not None:
        return round(depth, 1)
    suggested = _to_float(row.get("ai_suggested_depth_cm"))
    if suggested is not None and expected_flood == 1:
        return round(suggested, 1)
    return None


def build_manifest(source_path, output_path):
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    output_rows = []
    for row in rows:
        expected_flood = _resolve_expected_flood(row)
        expected_depth = _resolve_expected_depth(row, expected_flood)
        scene_type = _infer_scene_type(row, expected_flood)
        output_rows.append(
            {
                "image_path": (row.get("image_path") or "").replace("\\", "/"),
                "expected_flood": "" if expected_flood is None else str(expected_flood),
                "expected_depth_cm": "" if expected_depth is None else f"{expected_depth:.1f}",
                "scene_type": scene_type,
            }
        )

    existing_paths = {row["image_path"] for row in output_rows}
    repo_root = output_path.parent.parent.resolve()
    hard_negatives_dir = repo_root / "test_images" / "hard_negatives"
    if hard_negatives_dir.exists():
        for image_path in sorted(hard_negatives_dir.glob("*.jpg")):
            relative = str(image_path.relative_to(repo_root)).replace("\\", "/")
            if relative in existing_paths:
                continue
            name = image_path.name.lower()
            if "barren" in name:
                scene_type = "barren"
            else:
                scene_type = "urban"
            output_rows.append(
                {
                    "image_path": relative,
                    "expected_flood": "0",
                    "expected_depth_cm": "",
                    "scene_type": scene_type,
                }
            )
            existing_paths.add(relative)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "expected_flood", "expected_depth_cm", "scene_type"],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    labeled = sum(1 for row in output_rows if row["expected_flood"] in ("0", "1"))
    depth_labeled = sum(1 for row in output_rows if row["expected_depth_cm"] != "")
    barren_labeled = sum(1 for row in output_rows if row["scene_type"] == "barren")
    print(f"Manifest written: {output_path}")
    print(
        {
            "rows": len(output_rows),
            "expected_flood_labeled": labeled,
            "expected_depth_labeled": depth_labeled,
            "barren_rows": barren_labeled,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Create enterprise manifest for readiness evaluation.")
    parser.add_argument(
        "--source",
        default="test_images/evaluation_manifest_ai_seeded.csv",
        help="Source manifest with AI-assisted columns",
    )
    parser.add_argument(
        "--output",
        default="test_images/evaluation_manifest_labeled.csv",
        help="Output enterprise manifest path",
    )
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    output_path = Path(args.output).resolve()
    if not source_path.exists():
        raise SystemExit(f"Source manifest not found: {source_path}")
    build_manifest(source_path, output_path)


if __name__ == "__main__":
    main()
