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
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Flood Depth Estimator V6</title>
          <style>
            :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; color: #17324d; background: #f3f8fd; }
            * { box-sizing: border-box; }
            body { margin: 0; min-height: 100vh; }
            button, input { font: inherit; }
            button { cursor: pointer; }
            button:disabled { cursor: wait; opacity: .7; }
            .shell { max-width: 1160px; margin: 0 auto; padding: 52px 24px 72px; }
            .eyebrow { margin: 0 0 10px; color: #2472b6; font-size: .78rem; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; }
            h1 { margin: 0; font-size: clamp(2rem, 4vw, 3.1rem); line-height: 1.1; letter-spacing: -.045em; }
            .intro { margin: 14px 0 36px; max-width: 650px; color: #60758b; line-height: 1.6; }
            .grid { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 24px; align-items: stretch; }
            .panel { min-width: 0; padding: 30px; background: #fff; border: 1px solid #dceaf6; border-radius: 22px; box-shadow: 0 18px 48px rgba(25, 73, 116, .07); }
            .panel h2 { margin: 0 0 8px; font-size: 1.25rem; letter-spacing: -.02em; }
            .hint { margin: 0 0 24px; color: #6a7d90; font-size: .94rem; line-height: 1.55; }
            .dropzone { display: flex; min-height: 252px; flex-direction: column; align-items: center; justify-content: center; gap: 12px; padding: 24px; border: 2px dashed #a9c9e7; border-radius: 17px; background: #f7fbff; text-align: center; cursor: pointer; transition: border-color .2s, background .2s; }
            .dropzone:hover, .dropzone.is-dragging, .dropzone:focus-visible { border-color: #287bcc; background: #eef7ff; outline: none; }
            .upload-icon { display: grid; place-items: center; width: 58px; height: 58px; border-radius: 18px; background: #e4f1fc; color: #1873c8; font-size: 1.7rem; }
            .drop-title { font-weight: 750; }
            .drop-subtitle, .file-name { color: #71869a; font-size: .88rem; }
            #image { position: absolute; width: 1px; height: 1px; opacity: 0; }
            .preview-wrap { position: relative; margin-top: 18px; padding: 10px; border: 1px solid #dceaf6; border-radius: 16px; background: #f7fbff; }
            .preview-wrap img { display: block; width: 100%; max-height: 270px; object-fit: contain; border-radius: 10px; }
            .preview-footer { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 10px; }
            .file-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
            .text-button { border: 0; background: none; color: #146fbf; font-weight: 700; white-space: nowrap; }
            .primary-button { width: 100%; margin-top: 22px; padding: 15px 20px; border: 0; border-radius: 12px; background: #1677cf; color: white; font-weight: 750; box-shadow: 0 8px 18px rgba(22,119,207,.19); }
            .primary-button:hover:not(:disabled) { background: #0968bd; }
            .error { margin: 14px 0 0; color: #ae3030; font-size: .92rem; line-height: 1.5; }
            .loading { display: flex; align-items: center; gap: 10px; margin-top: 16px; color: #416886; font-size: .92rem; }
            .spinner { width: 18px; height: 18px; border: 2px solid #c8e2f8; border-top-color: #1677cf; border-radius: 50%; animation: spin .75s linear infinite; }
            @keyframes spin { to { transform: rotate(360deg); } }
            .result-panel { display: flex; flex-direction: column; }
            .result-body { display: flex; min-height: 300px; flex: 1; flex-direction: column; align-items: flex-start; justify-content: center; padding: 22px 0; }
            .result-label { margin: 0 0 13px; color: #52728f; font-weight: 700; }
            .result-value { margin: 0; color: #103d67; font-size: clamp(3.5rem, 7vw, 5.4rem); font-weight: 800; letter-spacing: -.07em; line-height: 1; }
            .result-value span { color: #397fb9; font-size: .42em; letter-spacing: -.02em; }
            .result-note { max-width: 370px; margin: 18px 0 0; color: #60758b; line-height: 1.5; }
            .result-source { margin: 12px 0 0; color: #52728f; font-size: .86rem; }
            .secondary-button { align-self: flex-start; padding: 11px 17px; border: 1px solid #bdd9ef; border-radius: 10px; background: #f4faff; color: #166db8; font-weight: 750; }
            .secondary-button:hover { background: #e6f3fe; }
            [hidden] { display: none !important; }
            .modal-backdrop { position: fixed; inset: 0; z-index: 10; display: grid; place-items: center; padding: 20px; background: rgba(15,39,63,.55); }
            .modal { display: flex; flex-direction: column; width: min(900px, 100%); max-height: min(82vh, 850px); border-radius: 18px; background: white; box-shadow: 0 24px 70px rgba(8,28,46,.28); }
            .modal-head { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 20px 24px; border-bottom: 1px solid #e2edf7; }
            .modal-head h2 { margin: 0; font-size: 1.15rem; }
            .modal-actions { display: flex; gap: 10px; }
            .modal-actions button { padding: 8px 12px; border: 1px solid #c8deef; border-radius: 9px; background: #f5faff; color: #165f9c; font-weight: 700; }
            .json-view { flex: 1; min-height: 0; margin: 0; padding: 22px 24px; overflow: auto; background: #f8fbfe; color: #27435b; font: .84rem/1.6 Consolas, Monaco, monospace; white-space: pre; }
            @media (max-width: 760px) { .shell { padding: 28px 16px 44px; } .grid { grid-template-columns: 1fr; } .panel { padding: 22px; } .result-body { min-height: 240px; } .modal-head { align-items: flex-start; flex-direction: column; } }
            @media (prefers-reduced-motion: reduce) { .spinner { animation: none; } }
          </style>
        </head>
        <body>
          <main class="shell">
            <p class="eyebrow">Image analysis</p>
            <h1>Flood Depth Estimator V6</h1>
            <p class="intro">Upload a flood scene to view its estimated water depth. Your complete prediction data is available in the JSON view.</p>
            <div class="grid">
              <section class="panel" aria-labelledby="upload-heading">
                <h2 id="upload-heading">Upload Image</h2>
                <p class="hint">Choose an image or drop it below to begin.</p>
                <form id="upload-form" action="/predict" method="post" enctype="multipart/form-data">
                  <label class="dropzone" id="dropzone" for="image" tabindex="0" role="button" aria-label="Choose or drop an image">
                    <span class="upload-icon" aria-hidden="true">↑</span>
                    <span class="drop-title">Drag and drop an image here</span>
                    <span class="drop-subtitle">or click to browse</span>
                  </label>
                  <input id="image" type="file" name="image" accept="image/*" required>
                  <div class="preview-wrap" id="preview-wrap" hidden>
                    <img id="preview" alt="Selected image preview">
                    <div class="preview-footer"><span class="file-name" id="file-name"></span><button class="text-button" id="remove-image" type="button">Remove image</button></div>
                  </div>
                  <button class="primary-button" id="analyze-button" type="submit">Analyze Image</button>
                  <div class="loading" id="loading" role="status" hidden><span class="spinner" aria-hidden="true"></span>Analyzing image…</div>
                  <p class="error" id="upload-error" role="alert" hidden></p>
                </form>
              </section>
              <section class="panel result-panel" aria-labelledby="result-heading">
                <h2 id="result-heading">Estimated Water Depth</h2>
                <p class="hint">Your result will appear here after analysis.</p>
                <div class="result-body" aria-live="polite">
                  <p class="result-label" id="result-status">Awaiting image</p>
                  <p class="result-value" id="result-value">—</p>
                  <p class="result-note" id="result-comment">Upload an image to get started.</p>
                  <p class="result-source" id="result-source" hidden></p>
                </div>
                <button class="secondary-button" id="view-json" type="button" disabled>View JSON</button>
              </section>
            </div>
          </main>
          <div class="modal-backdrop" id="json-backdrop" hidden>
            <div class="modal" role="dialog" aria-modal="true" aria-labelledby="json-heading">
              <div class="modal-head"><h2 id="json-heading">Complete JSON response</h2><div class="modal-actions"><button id="copy-json" type="button">Copy JSON</button><button id="close-json" type="button">Close</button></div></div>
              <pre class="json-view" id="json-content" tabindex="0"></pre>
            </div>
          </div>
          <script>
            (() => {
              const byId = id => document.getElementById(id);
              const form = byId("upload-form");
              const input = byId("image");
              const dropzone = byId("dropzone");
              const previewWrap = byId("preview-wrap");
              const preview = byId("preview");
              const errorBox = byId("upload-error");
              const backdrop = byId("json-backdrop");
              let selectedFile = null;
              let previewUrl = null;
              let lastResponse = null;
              let requestNumber = 0;

              const clearResult = () => {
                lastResponse = null;
                byId("view-json").disabled = true;
                byId("result-status").textContent = "Awaiting image";
                byId("result-value").textContent = "—";
                byId("result-comment").textContent = "Upload an image to get started.";
                byId("result-source").hidden = true;
              };
              const setFile = file => {
                requestNumber += 1;
                if (previewUrl) URL.revokeObjectURL(previewUrl);
                selectedFile = file || null;
                input.required = !selectedFile;
                previewUrl = selectedFile ? URL.createObjectURL(selectedFile) : null;
                previewWrap.hidden = !selectedFile;
                preview.src = previewUrl || "";
                byId("file-name").textContent = selectedFile ? selectedFile.name : "";
                errorBox.hidden = true;
                byId("analyze-button").disabled = false;
                byId("loading").hidden = true;
                clearResult();
              };
              input.addEventListener("change", () => setFile(input.files[0] || null));
              byId("remove-image").addEventListener("click", () => { input.value = ""; setFile(null); });
              dropzone.addEventListener("keydown", event => {
                if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); }
              });
              for (const name of ["dragenter", "dragover"]) dropzone.addEventListener(name, event => {
                event.preventDefault(); dropzone.classList.add("is-dragging");
              });
              for (const name of ["dragleave", "drop"]) dropzone.addEventListener(name, event => {
                event.preventDefault(); dropzone.classList.remove("is-dragging");
              });
              dropzone.addEventListener("drop", event => {
                const file = event.dataTransfer && event.dataTransfer.files[0];
                if (file) setFile(file);
              });
              const showError = message => {
                byId("result-status").textContent = "Analysis failed";
                byId("result-value").textContent = "—";
                byId("result-comment").textContent = message;
                byId("result-source").hidden = true;
                errorBox.textContent = message;
                errorBox.hidden = false;
              };
              const showResult = result => {
                const value = result.application_final_depth_cm;
                const noReference = result.skip_reason === "no_valid_reference_object_detected";
                if (typeof value === "number" && Number.isFinite(value)) {
                  byId("result-value").textContent = value.toFixed(2) + " cm";
                } else if (noReference && value === null) {
                  byId("result-value").textContent = "N/A";
                } else {
                  throw new Error("The response did not contain a usable depth.");
                }
                byId("result-status").textContent = result.prediction_status || "Estimated Water Depth";
                byId("result-comment").textContent = result.comment || "";
                const source = byId("result-source");
                source.textContent = result.decision_source ? "Source: " + result.decision_source : "";
                source.hidden = !result.decision_source;
                lastResponse = result;
                byId("view-json").disabled = false;
              };
              form.addEventListener("submit", async event => {
                event.preventDefault();
                if (!selectedFile) { showError("Choose an image before analyzing."); return; }
                const currentRequest = ++requestNumber;
                byId("analyze-button").disabled = true;
                byId("loading").hidden = false;
                errorBox.hidden = true;
                clearResult();
                byId("result-status").textContent = "Analyzing image";
                byId("result-comment").textContent = "Please wait while the image is processed.";
                try {
                  const body = new FormData(form);
                  body.set("image", selectedFile, selectedFile.name);
                  const response = await fetch("/predict", {method: "POST", body});
                  const result = await response.json();
                  if (currentRequest !== requestNumber) return;
                  if (!response.ok) throw new Error(result.error || "Analysis failed. Please try another image.");
                  showResult(result);
                } catch (error) {
                  if (currentRequest === requestNumber) showError(error.message || "Analysis failed. Please try again.");
                } finally {
                  if (currentRequest === requestNumber) {
                    byId("analyze-button").disabled = false;
                    byId("loading").hidden = true;
                  }
                }
              });
              const closeModal = () => { backdrop.hidden = true; byId("view-json").focus(); };
              byId("view-json").addEventListener("click", () => {
                if (!lastResponse) return;
                byId("json-content").textContent = JSON.stringify(lastResponse, null, 2);
                byId("copy-json").textContent = "Copy JSON";
                backdrop.hidden = false;
                byId("close-json").focus();
              });
              byId("close-json").addEventListener("click", closeModal);
              backdrop.addEventListener("click", event => { if (event.target === backdrop) closeModal(); });
              document.addEventListener("keydown", event => { if (event.key === "Escape" && !backdrop.hidden) closeModal(); });
              byId("copy-json").addEventListener("click", async () => {
                try {
                  await navigator.clipboard.writeText(byId("json-content").textContent);
                  byId("copy-json").textContent = "Copied";
                  setTimeout(() => { byId("copy-json").textContent = "Copy JSON"; }, 1500);
                } catch (_) { byId("copy-json").textContent = "Copy unavailable"; }
              });
            })();
          </script>
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
            from src.v6_application_review import review_v6_result
            return jsonify(review_v6_result(result, image_bytes, image_file.filename, get_v6_pipeline()))
        except Exception as exc:
            return jsonify({"error": f"V6 prediction failed: {exc}"}), 500

    @app.get("/status")
    def status():
        return jsonify({"status": "ready", "model": model_path})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
