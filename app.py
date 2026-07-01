"""
Flood Depth Estimator — Local Test Server
Simple Flask server for testing the water-aware trained model.

Usage:
    pip install flask torch torchvision pillow
    python app.py
    Open http://localhost:5000 in your browser
"""

import io
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# Model setup
# ─────────────────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "models" / "best_flood_model_water_aware.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model() -> nn.Module:
    """EfficientNet-B0 with scalar regression head — same arch used in training."""
    m = models.efficientnet_b0(weights=None)
    num_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    return m.to(DEVICE)


def load_model() -> nn.Module:
    model = build_model()
    if not MODEL_PATH.exists():
        logger.warning(f"Model file not found at {MODEL_PATH}. Using random weights.")
        model.eval()
        return model

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Log training metadata if available
    if isinstance(checkpoint, dict):
        epoch = checkpoint.get("epoch", "?")
        val_loss = checkpoint.get("val_loss", checkpoint.get("best_val_loss", "?"))
        logger.info(f"✅ Loaded water-aware model — epoch {epoch}, val_loss {val_loss}")
    return model


def depth_to_severity(depth_cm: float) -> dict:
    if depth_cm < 5:
        return {"level": "SAFE", "label": "No significant flooding", "color": "#22c55e"}
    elif depth_cm < 20:
        return {"level": "LOW", "label": "Minor flooding", "color": "#eab308"}
    elif depth_cm < 50:
        return {"level": "MEDIUM", "label": "Moderate flooding", "color": "#f97316"}
    elif depth_cm < 80:
        return {"level": "HIGH", "label": "High flooding", "color": "#ef4444"}
    else:
        return {"level": "CRITICAL", "label": "Severe / dangerous flooding", "color": "#7f1d1d"}


_MODEL = load_model()
logger.info(f"Model ready on {DEVICE}")

# ─────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB

UPLOAD_FORM = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Flood Depth Estimator</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 640px; margin: 60px auto; padding: 0 20px; background: #f8fafc; }
    h1 { color: #1e3a5f; }
    .card { background: #fff; border-radius: 12px; padding: 28px; box-shadow: 0 2px 12px rgba(0,0,0,.08); }
    input[type=file] { display: block; margin: 16px 0; }
    button { background: #1e3a5f; color: #fff; border: none; padding: 10px 28px; border-radius: 8px; cursor: pointer; font-size: 16px; }
    button:hover { background: #2563eb; }
    #result { margin-top: 24px; padding: 20px; border-radius: 10px; display: none; }
    .depth { font-size: 2.4rem; font-weight: 700; }
    .label { font-size: 1.1rem; margin-top: 6px; }
    #preview { max-width: 100%; border-radius: 8px; margin-top: 16px; display: none; }
    .meta { font-size: .85rem; color: #64748b; margin-top: 10px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>🌊 Flood Depth Estimator</h1>
    <p>Upload a flood image to predict water depth using the water-aware trained model.</p>
    <form id="form" enctype="multipart/form-data">
      <input type="file" id="file" name="image" accept="image/*" required>
      <img id="preview">
      <button type="submit">Estimate Depth</button>
    </form>
    <div id="result"></div>
  </div>
  <script>
    document.getElementById('file').addEventListener('change', function() {
      const reader = new FileReader();
      reader.onload = e => {
        const img = document.getElementById('preview');
        img.src = e.target.result;
        img.style.display = 'block';
      };
      reader.readAsDataURL(this.files[0]);
    });

    document.getElementById('form').addEventListener('submit', async function(e) {
      e.preventDefault();
      const btn = this.querySelector('button');
      btn.textContent = 'Analyzing…';
      btn.disabled = true;
      const fd = new FormData(this);
      try {
        const resp = await fetch('/predict', { method: 'POST', body: fd });
        const data = await resp.json();
        const r = document.getElementById('result');
        if (data.error) {
          r.style.background = '#fee2e2';
          r.innerHTML = '<b>Error:</b> ' + data.error;
        } else {
          r.style.background = data.severity.color + '22';
          r.style.border = '2px solid ' + data.severity.color;
          r.innerHTML = `
            <div class="depth" style="color:${data.severity.color}">${data.depth_cm} cm</div>
            <div class="label"><b>${data.severity.level}</b> — ${data.severity.label}</div>
            <div class="meta">Confidence: ${(data.confidence * 100).toFixed(1)}% &nbsp;|&nbsp; Device: ${data.device}</div>
          `;
        }
        r.style.display = 'block';
      } catch(err) {
        document.getElementById('result').innerHTML = 'Request failed: ' + err;
        document.getElementById('result').style.display = 'block';
      } finally {
        btn.textContent = 'Estimate Depth';
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(UPLOAD_FORM)


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Cannot open image: {e}"}), 400

    with torch.no_grad():
        tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)
        output = _MODEL(tensor)
        depth_norm = output.squeeze().item()

    depth_cm = round(depth_norm * 100.0, 2)
    confidence = round(min(depth_norm * 1.25, 1.0), 4)
    severity = depth_to_severity(depth_cm)

    logger.info(f"Predicted {depth_cm} cm ({severity['level']}) for {file.filename}")

    return jsonify({
        "depth_cm": depth_cm,
        "depth_normalized": round(depth_norm, 6),
        "confidence": confidence,
        "severity": severity,
        "model": str(MODEL_PATH.name),
        "device": str(DEVICE),
    })


@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "model": str(MODEL_PATH.name),
        "model_loaded": MODEL_PATH.exists(),
        "device": str(DEVICE),
    })


if __name__ == "__main__":
    logger.info("🌊 Starting Flood Depth Estimator at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
