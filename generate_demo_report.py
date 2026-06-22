import base64
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "flood_project.settings")

import django

django.setup()

from django.conf import settings

from flood_api.models import FloodInundationTelemetry
from flood_api.services.map_payload import build_dashboard_map_points
from flood_api.tasks import process_and_refine_telemetry


def _collect_demo_images(limit=10):
    test_root = Path(settings.BASE_DIR) / "test_images"
    files = []
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(test_root.rglob(pattern))
    files = sorted({f.resolve() for f in files if f.is_file()})
    return files[:limit]


def _img_to_data_uri(path: Path):
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _render_report(rows, output_path: Path):
    cards_html = []
    for idx, row in enumerate(rows):
        point_json = json.dumps(row["map_point"], indent=2)
        point_json_escaped = (
            point_json.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        cards_html.append(
            f"""
            <section class="card">
              <div class="card-title">Image {idx + 1}: {row["image_name"]}</div>
              <div class="verdict {row["map_point"].get("response_verdict","Needs Review").lower().replace(" ", "-")}">
                Response Verdict: {row["map_point"].get("response_verdict", "Needs Review")}
                <span class="meta">(source: {row["map_point"].get("verdict_source", "unverified")})</span>
              </div>
              <div class="split">
                <div class="pane">
                  <h3>Uploaded Image</h3>
                  <img class="preview" src="{row["image_data_uri"]}" alt="{row["image_name"]}">
                </div>
                <div class="pane">
                  <h3>Output (Bengaluru Map + JSON)</h3>
                  <div id="map-{idx}" class="map"></div>
                  <pre>{point_json_escaped}</pre>
                </div>
              </div>
            </section>
            """
        )

    points_js = json.dumps([r["map_point"] for r in rows])
    maps_init = []
    for idx in range(len(rows)):
        maps_init.append(
            f"""
            (function() {{
              const p = points[{idx}];
              const map = L.map('map-{idx}').setView([p.latitude, p.longitude], 12);
              L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                maxZoom: 19,
                attribution: '&copy; OpenStreetMap contributors'
              }}).addTo(map);
              const color = intensityColor[p.intensity_scale] || '#3b82f6';
              L.circleMarker([p.latitude, p.longitude], {{
                radius: 10,
                color: '#0f172a',
                weight: 1,
                fillColor: color,
                fillOpacity: 0.95
              }}).addTo(map).bindPopup(
                `<b>${{p.location_name}}</b><br/>Depth: ${{p.depth_cm}} cm<br/>Scale: ${{p.intensity_scale}}/5`
              ).openPopup();
            }})();
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Flood System Demo Report (10 Images)</title>
  <link
    rel="stylesheet"
    href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
    crossorigin=""
  />
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 18px; }}
    .header {{ background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 16px; margin-bottom: 14px; }}
    .header h1 {{ margin: 0 0 8px 0; }}
    .header p {{ margin: 0; color: #94a3b8; }}
    .card {{ background: #111827; border: 1px solid #334155; border-radius: 12px; padding: 12px; margin-bottom: 14px; }}
    .card-title {{ font-weight: 700; margin-bottom: 10px; }}
    .split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .pane {{ background: #0b1220; border: 1px solid #334155; border-radius: 10px; padding: 10px; }}
    .verdict {{ margin: 0 0 10px 0; padding: 8px 10px; border-radius: 8px; font-weight: 700; }}
    .verdict.correct {{ background: rgba(34,197,94,0.2); border: 1px solid rgba(34,197,94,0.6); }}
    .verdict.incorrect {{ background: rgba(239,68,68,0.2); border: 1px solid rgba(239,68,68,0.7); }}
    .verdict.needs-review {{ background: rgba(245,158,11,0.2); border: 1px solid rgba(245,158,11,0.7); }}
    .verdict .meta {{ font-weight: 500; color: #cbd5e1; margin-left: 8px; font-size: 12px; }}
    .preview {{ width: 100%; max-height: 420px; object-fit: contain; border-radius: 8px; border: 1px solid #334155; background: #020617; }}
    .map {{ width: 100%; height: 260px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 8px; }}
    pre {{ margin: 0; background: #020617; border: 1px solid #334155; border-radius: 8px; padding: 10px; max-height: 220px; overflow: auto; font-size: 12px; color: #cbd5e1; }}
    h3 {{ margin: 0 0 8px 0; }}
    @media (max-width: 1000px) {{ .split {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>Enterprise Flood Monitoring Demo Report</h1>
      <p>Generated: {datetime.now().isoformat(timespec="seconds")} | Samples: {len(rows)} | Layout: Image (left) + Bengaluru map + JSON output (right)</p>
    </div>
    {''.join(cards_html)}
  </div>
  <script
    src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
    crossorigin=""
  ></script>
  <script>
    const points = {points_js};
    const intensityColor = {{
      1: "#22c55e",
      2: "#84cc16",
      3: "#facc15",
      4: "#f97316",
      5: "#ef4444"
    }};
    {''.join(maps_init)}
  </script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


def main():
    images = _collect_demo_images(limit=10)
    if len(images) < 10:
        raise RuntimeError(f"Need at least 10 images under test_images/. Found {len(images)}")

    runtime_root = Path(getattr(settings, "RUNTIME_ROOT", Path(r"E:\flood_runtime")))
    tmp_dir = runtime_root / "tmp_uploads"
    report_dir = runtime_root / "reports"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, img in enumerate(images, start=1):
        temp_copy = tmp_dir / f"demo_{uuid4().hex}_{img.name}"
        shutil.copy2(img, temp_copy)

        result = process_and_refine_telemetry.run(
            image_filepath=str(temp_copy),
            filename=img.name,
            external_context="Enterprise Demo Report Generation",
            camera_id=f"demo_report_cam_{idx:02d}",
        )
        if result.get("status") != "success":
            continue

        record = FloodInundationTelemetry.objects.select_related("camera").get(id=result["record_id"])
        map_point = build_dashboard_map_points([record])[0]
        rows.append(
            {
                "image_name": img.name,
                "image_data_uri": _img_to_data_uri(img),
                "map_point": map_point,
            }
        )

    if len(rows) == 0:
        raise RuntimeError("No successful evaluations were produced.")

    out_path = report_dir / f"flood_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    _render_report(rows, out_path)
    print(str(out_path))


if __name__ == "__main__":
    main()
