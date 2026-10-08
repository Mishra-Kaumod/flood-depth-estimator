"""Lightweight Flask web interface for the cleaned flood analysis project."""

from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request
from pydantic import ValidationError

from src.v6_inference import create_v6_pipeline, load_v6_rgb, v6_depth_payload


def create_app(model_path: str = "severity_model.pth") -> Flask:
    app = Flask(__name__)
    api_service = None
    v6_pipeline = None

    def get_v6_pipeline():
        nonlocal v6_pipeline
        if v6_pipeline is None:
            v6_pipeline = create_v6_pipeline()
        return v6_pipeline

    def get_api_service():
        nonlocal api_service
        if api_service is None:
            from src.api_service import FloodApiService
            api_service = FloodApiService()
        return api_service

    @app.get("/")
    def index():
        return render_template_string("""
        <!doctype html>
        <html>
          <head>
            <meta charset="utf-8">
            <title>Flood Depth Estimator – V6</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 2rem; line-height: 1.5; }
              .card { max-width: 720px; padding: 1.5rem; border: 1px solid #d0d7de; border-radius: 12px; }
              button { padding: 0.6rem 1rem; margin-top: 0.75rem; }
              pre { background: #f6f8fa; padding: 1rem; border-radius: 8px; overflow-x: auto; }
            </style>
          </head>
          <body>
            <div class="card">
              <h1>Flood Depth Estimator – V6</h1>
              <p>Select an image to estimate flood depth with V6. Results are shown in cm.</p>
              <form id="upload-form" action="/predict" method="post" enctype="multipart/form-data">
                <label for="image">Select an image</label><br>
                <input id="image" type="file" name="image" accept="image/*" required>
                <br>
                <button type="submit">Analyze Image</button>
              </form>
              <p id="depth-result" aria-live="polite"></p>
              <script>
                document.getElementById("upload-form").addEventListener("submit", async (event) => {
                  event.preventDefault();
                  const output = document.getElementById("depth-result");
                  output.textContent = "Analyzing…";
                  try {
                    const response = await fetch("/predict", {method: "POST", body: new FormData(event.target)});
                    const result = await response.json();
                    if (!response.ok) throw new Error(result.error || "Analysis failed");
                    output.textContent = result.final_shadow_depth_cm === null
                      ? "Flood depth unavailable"
                      : "Flood depth: " + result.final_shadow_depth_cm + " cm";
                  } catch (error) {
                    output.textContent = error.message;
                  }
                });
              </script>
            </div>
          </body>
        </html>
        """)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "flood-analysis", "architecture": "V6"})

    @app.post("/api/v1/estimate")
    @app.post("/api/v1/estimate/")
    def camera_upload_api():
        try:
            metadata = {}
            image_bytes = None
            filename = "camera_upload.jpg"

            if request.is_json:
                payload = request.get_json() or {}
                import base64

                image_b64 = payload.get("image_b64")
                if not image_b64:
                    return jsonify({"status": "failed", "error": "image_b64 is required"}), 400
                image_bytes = base64.b64decode(image_b64, validate=True)
                metadata = payload.get("metadata") or {}
                camera_id = payload.get("camera_id", "intersection_01")
                latitude = float(payload.get("latitude"))
                longitude = float(payload.get("longitude"))
                location_name = payload.get("location_name")
                filename = payload.get("filename", filename)
            else:
                uploaded_file = request.files.get("image")
                if uploaded_file is None:
                    return jsonify({"status": "failed", "error": "No image payload"}), 400
                image_bytes = uploaded_file.read()
                filename = uploaded_file.filename or filename
                camera_id = request.form.get("camera_id", "intersection_01")
                latitude = float(request.form.get("latitude"))
                longitude = float(request.form.get("longitude"))
                location_name = request.form.get("location_name")
                metadata = {"context": request.form.get("context", "")}

            response = get_api_service().process_camera_upload(
                image_bytes=image_bytes,
                filename=filename,
                camera_id=camera_id,
                latitude=latitude,
                longitude=longitude,
                location_name=location_name,
                metadata=metadata,
            )
            return jsonify(response), 200
        except (KeyError, TypeError, ValueError, ValidationError, OSError) as exc:
            return jsonify({"status": "failed", "error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"status": "error", "error": f"V6 prediction failed: {exc}"}), 500

    @app.get("/api/v1/temporal/<camera_id>")
    @app.get("/api/v1/temporal/<camera_id>/")
    def get_temporal_sequence(camera_id):
        sequence = get_api_service().latest_temporal_sequence(camera_id)
        if sequence is None:
            return jsonify(
                {
                    "status": "no_data",
                    "message": f"No temporal sequences found for camera {camera_id}",
                    "camera_id": camera_id,
                }
            ), 404
        return jsonify({"status": "success", "sequence": sequence})

    @app.post("/api/v1/temporal/<camera_id>/analyze")
    @app.post("/api/v1/temporal/<camera_id>/analyze/")
    def trigger_temporal_analysis(camera_id):
        time_window = int(request.values.get("time_window", 15))
        result = get_api_service().trigger_temporal_analysis(
            camera_id=camera_id,
            time_window_minutes=time_window,
        )
        return jsonify(result), 202

    @app.get("/api/v1/camera/<camera_id>/stats")
    @app.get("/api/v1/camera/<camera_id>/stats/")
    def get_camera_stats(camera_id):
        hours = int(request.args.get("hours", 24))
        return jsonify({"status": "success", **get_api_service().camera_stats(camera_id, hours)})

    @app.get("/api/v1/telemetry")
    @app.get("/api/v1/telemetry/")
    def recent_telemetry():
        limit = int(request.args.get("limit", 20))
        camera_id = request.args.get("camera_id")
        return jsonify(
            {
                "status": "success",
                "records": get_api_service().recent_telemetry(limit=limit, camera_id=camera_id),
            }
        )

    @app.post("/predict")
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        image_bytes = image_file.read()
        if not image_bytes:
            return jsonify({"error": "Empty image"}), 400

        from PIL import UnidentifiedImageError

        try:
            image_rgb = load_v6_rgb(image_bytes)
        except (UnidentifiedImageError, OSError, ValueError):
            return jsonify({"error": "Could not decode image"}), 400

        try:
            result = get_v6_pipeline().predict(image_rgb)
            return jsonify(v6_depth_payload(result))
        except Exception as exc:
            return jsonify({"error": f"V6 prediction failed: {exc}"}), 500

    @app.get("/status")
    def status():
        return jsonify({"status": "ready", "model": model_path})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
