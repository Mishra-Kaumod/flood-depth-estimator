"""Lightweight Flask web interface for the cleaned flood analysis project."""

from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request
from pydantic import ValidationError


def create_app(model_path: str = "severity_model.pth") -> Flask:
    app = Flask(__name__)
    api_service = None

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
            <title>Flood Depth Estimator</title>
            <style>
              body { font-family: Arial, sans-serif; margin: 2rem; line-height: 1.5; }
              .card { max-width: 720px; padding: 1.5rem; border: 1px solid #d0d7de; border-radius: 12px; }
              button { padding: 0.6rem 1rem; margin-top: 0.75rem; }
              pre { background: #f6f8fa; padding: 1rem; border-radius: 8px; overflow-x: auto; }
            </style>
          </head>
          <body>
            <div class="card">
              <h1>Flood Depth Estimator</h1>
              <p>Select an image to run the cleaned flood analysis pipeline and inspect the result directly in the browser.</p>
              <form action="/predict" method="post" enctype="multipart/form-data">
                <label for="image">Select an image</label><br>
                <input id="image" type="file" name="image" accept="image/*" required>
                <br>
                <button type="submit">Analyze Image</button>
              </form>
            </div>
          </body>
        </html>
        """)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "flood-analysis", "model": model_path})

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
            return jsonify(response), 202
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            return jsonify({"status": "failed", "error": str(exc)}), 400
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

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

        import cv2
        import numpy as np

        np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Could not decode image"}), 400

        import base64
        image_b64 = base64.b64encode(image_bytes).decode("ascii")

        response = get_api_service().process_camera_upload(
            image_bytes=image_bytes,
            filename=image_file.filename,
            camera_id=request.form.get("camera_id", "web_camera"),
            latitude=float(request.form.get("latitude", 0.0)),
            longitude=float(request.form.get("longitude", 0.0)),
            location_name=request.form.get("location_name"),
            metadata={"context": request.form.get("context", "web_upload")},
        )

        result = response.get("result", {})
        payload = {
            "image_path": result.get("image_name"),
            "water_detected": result.get("estimated_depth_meters", 0) > 0,
            "final_flood_level": result.get("severity_label"),
            "depth_cm": round(result.get("estimated_depth_meters", 0.0) * 100.0, 2),
            "severity_name": result.get("severity_label"),
            "water_percentage": None,
            "water_confidence": round(result.get("confidence_score", 0.0), 4),
            "depth_method": result.get("method"),
            "depth_details": result.get("metadata", {}),
            "method_votes": {},
            "production_depth_cm": round(result.get("estimated_depth_meters", 0.0) * 100.0, 2),
            "production_method": result.get("method"),
            "production_action": result.get("action_trigger"),
            "production_trace": result.get("metadata", {}).get("pipeline_trace", []),
            "production_reference_estimate": None,
            "status": response.get("status"),
        }
        return jsonify(payload)

    @app.get("/status")
    def status():
        return jsonify({"status": "ready", "model": model_path})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
