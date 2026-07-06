"""
Flood Depth Estimator — Rich Web UI
3-section interface: image upload grid + live Bengaluru map + intensity results map
Prediction backend: Gemini vision endpoint, with reference-object CV fallback.
"""

import base64
import io
import json
import logging

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

from src.reference_depth_estimator import ReferenceDepthEstimator
from src.gemini_depth_estimator import GeminiDepthEstimator, GeminiError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BENGALURU_AREAS = [
    {"name": "Koramangala",    "lat": 12.9344, "lng": 77.6269},
    {"name": "Indiranagar",    "lat": 12.9784, "lng": 77.6408},
    {"name": "HSR Layout",     "lat": 12.9116, "lng": 77.6389},
    {"name": "Whitefield",     "lat": 12.9698, "lng": 77.7500},
    {"name": "Electronic City","lat": 12.8452, "lng": 77.6602},
    {"name": "Marathahalli",   "lat": 12.9591, "lng": 77.7006},
    {"name": "JP Nagar",       "lat": 12.9102, "lng": 77.5921},
    {"name": "Bellandur",      "lat": 12.9239, "lng": 77.6762},
    {"name": "Banashankari",   "lat": 12.9265, "lng": 77.5640},
    {"name": "Hebbal",         "lat": 13.0358, "lng": 77.5970},
    {"name": "Malleshwaram",   "lat": 13.0035, "lng": 77.5713},
    {"name": "Jayanagar",      "lat": 12.9299, "lng": 77.5838},
    {"name": "MG Road",        "lat": 12.9756, "lng": 77.6079},
    {"name": "Yelahanka",      "lat": 13.1005, "lng": 77.5963},
    {"name": "Sarjapur",       "lat": 12.8614, "lng": 77.7871},
]

# ─────────────────────────────────────────────────────────
# Prediction backends
# ─────────────────────────────────────────────────────────

def depth_to_severity(depth_cm: float) -> dict:
    if depth_cm < 5:
        return {"level": "SAFE",     "label": "No significant flooding",       "color": "#16a34a", "stage": 1}
    elif depth_cm < 20:
        return {"level": "LOW",      "label": "Minor flooding",                 "color": "#ca8a04", "stage": 2}
    elif depth_cm < 50:
        return {"level": "MEDIUM",   "label": "Moderate flooding",              "color": "#ea580c", "stage": 3}
    elif depth_cm < 80:
        return {"level": "HIGH",     "label": "High flood — avoid travel",      "color": "#dc2626", "stage": 4}
    else:
        return {"level": "CRITICAL", "label": "Severe / dangerous flooding",    "color": "#7f1d1d", "stage": 5}

_GEMINI_ESTIMATOR = GeminiDepthEstimator.from_env()
_REFERENCE_ESTIMATOR = ReferenceDepthEstimator()

if _GEMINI_ESTIMATOR.available:
    logger.info(f"✅ Gemini prediction backend ready — model {_GEMINI_ESTIMATOR.model_name}")
else:
    logger.warning(
        "🔀 GEMINI_API_KEY not set → predictions will use reference_object_cv fallback"
    )

# ─────────────────────────────────────────────────────────
# HTML Template
# ─────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Bengaluru Flood Depth Estimator</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#eef2f7;color:#1e293b;height:100vh;display:flex;flex-direction:column;overflow:hidden}

/* ── Header ── */
.hdr{background:linear-gradient(135deg,#1e3a5f 0%,#1d4ed8 100%);color:#fff;padding:10px 20px;display:flex;align-items:center;justify-content:space-between;flex-shrink:0;box-shadow:0 2px 10px rgba(0,0,0,.25);z-index:100}
.hdr h1{font-size:1.15rem;font-weight:700;display:flex;align-items:center;gap:8px}
.badge{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);padding:3px 12px;border-radius:20px;font-size:.78rem;display:flex;align-items:center;gap:6px}
.dot{width:8px;height:8px;border-radius:50%;background:#4ade80}

/* ── Views ── */
.view{flex:1;display:flex;overflow:hidden;transition:opacity .25s}

/* ── Upload View ── */
#v-upload{display:flex}

.upload-col{width:52%;background:#fff;display:flex;flex-direction:column;border-right:1px solid #e2e8f0;overflow:hidden}
.map-col{flex:1;display:flex;flex-direction:column}

.col-hdr{padding:12px 16px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;justify-content:space-between;flex-shrink:0}
.col-hdr h2{font-size:.95rem;font-weight:600;color:#1e3a5f;display:flex;align-items:center;gap:6px}
.col-hdr small{color:#64748b;font-size:.78rem}

/* Drop zone */
.drop-zone{margin:12px 14px;border:2px dashed #93c5fd;border-radius:10px;background:#eff6ff;cursor:pointer;transition:all .2s;flex-shrink:0}
.drop-zone:hover,.drop-zone.over{border-color:#2563eb;background:#dbeafe}
.dz-inner{padding:18px;text-align:center;pointer-events:none}
.dz-inner svg{color:#60a5fa;margin-bottom:6px}
.dz-inner p{font-size:.88rem;color:#3b82f6;font-weight:500}
.dz-inner span{font-size:.75rem;color:#94a3b8}

/* Image grid */
#img-grid{flex:1;overflow-y:auto;padding:6px 12px;display:flex;flex-direction:column;gap:8px}
.img-card{background:#f8fafc;border:2px solid #e2e8f0;border-radius:10px;padding:10px;display:flex;gap:10px;align-items:flex-start;transition:border-color .2s,box-shadow .2s;cursor:pointer}
.img-card:hover{border-color:#93c5fd}
.img-card.active{border-color:#2563eb;box-shadow:0 0 0 3px rgba(37,99,235,.15);background:#eff6ff}
.img-thumb{width:60px;height:60px;object-fit:cover;border-radius:6px;flex-shrink:0;background:#e2e8f0}
.img-meta{flex:1;min-width:0}
.img-name{font-size:.8rem;font-weight:600;color:#1e293b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:5px}
.img-controls{display:grid;grid-template-columns:1fr 1fr;gap:5px}
.img-controls select,.img-controls input{font-size:.75rem;padding:4px 6px;border:1px solid #cbd5e1;border-radius:5px;background:#fff;color:#374151;width:100%}
.img-controls select:focus,.img-controls input:focus{outline:none;border-color:#3b82f6}
.coord-row{grid-column:1/-1;display:grid;grid-template-columns:1fr 1fr;gap:5px}
.pin-num{width:22px;height:22px;border-radius:50%;background:#2563eb;color:#fff;font-size:.7rem;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:2px}
.rm-btn{background:none;border:none;cursor:pointer;color:#94a3b8;font-size:1rem;padding:2px;line-height:1;flex-shrink:0;margin-top:0}
.rm-btn:hover{color:#ef4444}

/* Footer buttons */
.upload-footer{padding:12px 14px;border-top:1px solid #e2e8f0;flex-shrink:0}
.analyze-btn{width:100%;padding:11px;background:#1d4ed8;color:#fff;border:none;border-radius:8px;font-size:.95rem;font-weight:600;cursor:pointer;display:flex;align-items:center;justify-content:center;gap:8px;transition:background .2s}
.analyze-btn:hover:not(:disabled){background:#1e40af}
.analyze-btn:disabled{background:#94a3b8;cursor:not-allowed}

/* Map */
#upload-map{flex:1;min-height:0}
.map-hint{font-size:.73rem;color:#64748b;padding:6px 14px;border-bottom:1px solid #e2e8f0;flex-shrink:0;text-align:center;background:#f8fafc}
.map-hint b{color:#2563eb}

/* ── Results View ── */
#v-results{display:none}
.results-sidebar{width:300px;background:#fff;border-right:1px solid #e2e8f0;display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
.r-hdr{padding:12px 16px;border-bottom:1px solid #e2e8f0;flex-shrink:0}
.back-btn{display:flex;align-items:center;gap:6px;background:none;border:1px solid #cbd5e1;padding:5px 10px;border-radius:6px;cursor:pointer;font-size:.8rem;color:#475569;margin-bottom:10px;width:100%;justify-content:center}
.back-btn:hover{background:#f1f5f9}
.r-hdr h2{font-size:.95rem;font-weight:600;color:#1e3a5f}
.r-hdr p{font-size:.75rem;color:#64748b;margin-top:2px}

/* Intensity legend */
.legend{display:flex;flex-direction:column;gap:3px;padding:10px 16px;border-bottom:1px solid #e2e8f0;flex-shrink:0}
.legend-row{display:flex;align-items:center;gap:8px;font-size:.8rem}
.legend-dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.legend-label{flex:1;color:#374151}
.legend-count{font-weight:700;color:#1e293b}

/* Results list */
#results-list{flex:1;overflow-y:auto;padding:8px}
.r-card{background:#f8fafc;border-radius:8px;padding:10px 12px;margin-bottom:7px;border-left:4px solid #e2e8f0;cursor:pointer;transition:background .15s}
.r-card:hover{background:#eff6ff}
.r-card-name{font-size:.82rem;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.r-card-depth{font-size:1.3rem;font-weight:700;line-height:1.1}
.r-card-level{font-size:.73rem;margin-top:1px}
.r-card-loc{font-size:.72rem;color:#64748b;margin-top:3px}

#results-map{flex:1}
</style>
</head>
<body>

<!-- Header -->
<header class="hdr">
  <h1>🌊 Bengaluru Flood Depth Estimator</h1>
  <div class="badge"><span class="dot" id="mdot"></span><span id="mstat">Loading model…</span></div>
</header>

<!-- ── UPLOAD VIEW ── -->
<div class="view" id="v-upload">

  <!-- Left: upload panel -->
  <div class="upload-col">
    <div class="col-hdr">
      <h2>📷 Flood Images</h2>
      <small id="img-counter">0 / 10 images</small>
    </div>

    <label for="file-in" class="drop-zone" id="drop-zone">
      <div class="dz-inner">
        <svg width="36" height="36" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"/>
        </svg>
        <p>Drag &amp; drop flood images</p>
        <span>Up to 10 images · JPG, PNG, WebP</span>
      </div>
    </label>
    <input type="file" id="file-in" accept="image/*" multiple style="position:fixed;top:-200px;left:-200px;width:1px;height:1px;opacity:0">

    <div id="img-grid"></div>

    <div class="upload-footer">
      <button class="analyze-btn" id="analyze-btn" disabled onclick="analyzeAll()">
        🔍 Analyze Flood Depths
      </button>
    </div>
  </div>

  <!-- Right: Bengaluru map -->
  <div class="map-col">
    <div class="col-hdr" style="border-bottom:none;padding-bottom:0">
      <h2>📍 Bengaluru Location Map</h2>
    </div>
    <div class="map-hint" id="map-hint">Select an image card, then <b>click on the map</b> to set its location</div>
    <div id="upload-map"></div>
  </div>
</div>

<!-- ── RESULTS VIEW ── -->
<div class="view" id="v-results">
  <div class="results-sidebar">
    <div class="r-hdr">
      <button class="back-btn" onclick="showUpload()">← Upload More Images</button>
      <h2>📊 Flood Analysis</h2>
      <p id="r-subtitle"></p>
    </div>
    <div class="legend" id="legend"></div>
    <div id="results-list"></div>
  </div>
  <div id="results-map"></div>
</div>

<script>
// ─── State ───────────────────────────────────────────────
const AREAS = {{ areas | safe }};
const STAGES = [
  {level:'SAFE',    color:'#16a34a', label:'No significant flooding', min:0,  max:5  },
  {level:'LOW',     color:'#ca8a04', label:'Minor flooding',           min:5,  max:20 },
  {level:'MEDIUM',  color:'#ea580c', label:'Moderate flooding',        min:20, max:50 },
  {level:'HIGH',    color:'#dc2626', label:'High flood — avoid travel',min:50, max:80 },
  {level:'CRITICAL',color:'#7f1d1d', label:'Severe / dangerous',       min:80, max:999},
];

let images = [];          // {id,file,name,dataUrl,lat,lng,area}
let selectedId = null;    // id of image being location-targeted
let uploadMap = null;
let uploadPins = {};      // {id: L.marker}
let resultsMap = null;
let nextId = 1;

// ─── Model Status ─────────────────────────────────────────
fetch('/health').then(r=>r.json()).then(d=>{
  const ok = d.status === 'ok' && d.model_loaded;
  if (d.warning === 'label_collapse') {
    document.getElementById('mdot').style.background = '#f87171';
    document.getElementById('mstat').textContent = '⚠️ Model needs retraining — trained on unlabeled data (outputs 0cm)';
    // Show a banner
    const banner = document.createElement('div');
    banner.style.cssText = 'background:#fef2f2;border:1.5px solid #fca5a5;color:#991b1b;padding:8px 16px;font-size:.82rem;text-align:center;flex-shrink:0';
    banner.innerHTML = '⚠️ <b>Label Collapse:</b> Current model was trained with all-zero labels and always predicts 0 cm. ' +
      'Retrain using the <b>Flood_Depth_Google_Colab.ipynb</b> — complete <b>Step 7</b> to create <code>labels.csv</code> before training.';
    document.querySelector('.hdr').insertAdjacentElement('afterend', banner);
  } else {
    document.getElementById('mdot').style.background = ok ? '#4ade80' : '#f87171';
    document.getElementById('mstat').textContent = ok
      ? `✅ ${d.model} · ${d.device}`
      : '⚠️ Model not loaded';
  }
}).catch(()=>{ document.getElementById('mstat').textContent = '❌ Server error'; });

// ─── Upload Map init ──────────────────────────────────────
window.addEventListener('load', () => {
  uploadMap = L.map('upload-map', {zoomControl: true}).setView([12.9716, 77.5946], 11);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors', maxZoom: 18
  }).addTo(uploadMap);

  uploadMap.on('click', e => {
    if (!selectedId) return;
    const {lat, lng} = e.latlng;
    setImageLocation(selectedId, lat.toFixed(5), lng.toFixed(5), 'Custom');
    clearSelection();
  });
});

// ─── Drop zone ────────────────────────────────────────────
const dz = document.getElementById('drop-zone');
dz.addEventListener('dragover', e => { e.preventDefault(); e.stopPropagation(); dz.classList.add('over'); });
dz.addEventListener('dragleave', e => { e.stopPropagation(); dz.classList.remove('over'); });
dz.addEventListener('drop', e => {
  e.preventDefault(); e.stopPropagation();
  dz.classList.remove('over');
  addFiles(e.dataTransfer.files);
});
const fileIn = document.getElementById('file-in');
fileIn.addEventListener('change', e => { if (e.target.files.length) addFiles(e.target.files); fileIn.value = ''; });

function addFiles(files) {
  for (const file of files) {
    if (images.length >= 10) break;
    if (!file.type.startsWith('image/')) continue;
    const id = nextId++;
    const reader = new FileReader();
    reader.onload = ev => {
      const area = AREAS[images.length % AREAS.length];
      images.push({ id, file, name: file.name, dataUrl: ev.target.result,
                    lat: area.lat.toFixed(5), lng: area.lng.toFixed(5), area: area.name });
      renderGrid();
      addPin(images[images.length - 1]);
      updateCounter();
    };
    reader.readAsDataURL(file);
  }
  document.getElementById('file-in').value = '';
}

// ─── Image grid render ────────────────────────────────────
function renderGrid() {
  const grid = document.getElementById('img-grid');
  grid.innerHTML = '';
  images.forEach((img, idx) => {
    const card = document.createElement('div');
    card.className = 'img-card' + (selectedId === img.id ? ' active' : '');
    card.dataset.id = img.id;
    card.innerHTML = `
      <div class="pin-num">${idx + 1}</div>
      <img class="img-thumb" src="${img.dataUrl}" alt="">
      <div class="img-meta">
        <div class="img-name" title="${img.name}">${img.name}</div>
        <div class="img-controls">
          <select onchange="onAreaChange(${img.id}, this.value)" style="grid-column:1/-1">
            <option value="">📍 Select area…</option>
            ${AREAS.map(a => `<option value="${a.name}" ${a.name===img.area?'selected':''}>${a.name}</option>`).join('')}
            <option value="Custom" ${img.area==='Custom'?'selected':''}>Custom (click map)</option>
          </select>
          <div class="coord-row">
            <input type="number" step="0.0001" placeholder="Latitude" value="${img.lat}"
                   onchange="onCoordChange(${img.id},'lat',this.value)">
            <input type="number" step="0.0001" placeholder="Longitude" value="${img.lng}"
                   onchange="onCoordChange(${img.id},'lng',this.value)">
          </div>
        </div>
      </div>
      <button class="rm-btn" onclick="removeImage(${img.id},event)" title="Remove">✕</button>
    `;
    card.addEventListener('click', e => {
      if (e.target.closest('.rm-btn') || e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
      toggleSelect(img.id);
    });
    grid.appendChild(card);
  });
  document.getElementById('analyze-btn').disabled = images.length === 0;
}

function updateCounter() {
  document.getElementById('img-counter').textContent = `${images.length} / 10 images`;
  document.getElementById('analyze-btn').disabled = images.length === 0;
}

function toggleSelect(id) {
  selectedId = (selectedId === id) ? null : id;
  document.getElementById('map-hint').innerHTML = selectedId
    ? `<b>Click on the map</b> to place pin for image ${images.findIndex(i=>i.id===selectedId)+1}`
    : `Select an image card, then <b>click on the map</b> to set its location`;
  uploadMap.getContainer().style.cursor = selectedId ? 'crosshair' : '';
  renderGrid();
}

function clearSelection() {
  selectedId = null;
  document.getElementById('map-hint').innerHTML = `Select an image card, then <b>click on the map</b> to set its location`;
  uploadMap.getContainer().style.cursor = '';
  renderGrid();
}

function onAreaChange(id, areaName) {
  const img = images.find(i => i.id === id);
  if (!img) return;
  if (areaName === 'Custom') { img.area = 'Custom'; renderGrid(); return; }
  const area = AREAS.find(a => a.name === areaName);
  if (area) { img.lat = area.lat.toFixed(5); img.lng = area.lng.toFixed(5); img.area = area.name; }
  renderGrid();
  addPin(img);
}

function onCoordChange(id, field, val) {
  const img = images.find(i => i.id === id);
  if (!img) return;
  img[field] = val; img.area = 'Custom';
  addPin(img);
}

function setImageLocation(id, lat, lng, area) {
  const img = images.find(i => i.id === id);
  if (!img) return;
  img.lat = lat; img.lng = lng; img.area = area;
  renderGrid();
  addPin(img);
}

function removeImage(id, e) {
  e.stopPropagation();
  images = images.filter(i => i.id !== id);
  if (uploadPins[id]) { uploadMap.removeLayer(uploadPins[id]); delete uploadPins[id]; }
  if (selectedId === id) clearSelection();
  renderGrid(); updateCounter();
}

// ─── Map pins (upload view) ───────────────────────────────
function makeIcon(num, color) {
  return L.divIcon({
    className: '',
    html: `<div style="width:28px;height:28px;border-radius:50%;background:${color};border:2px solid #fff;
                box-shadow:0 2px 6px rgba(0,0,0,.35);display:flex;align-items:center;justify-content:center;
                color:#fff;font-size:11px;font-weight:700">${num}</div>`,
    iconSize: [28, 28], iconAnchor: [14, 14]
  });
}

function addPin(img) {
  const lat = parseFloat(img.lat), lng = parseFloat(img.lng);
  if (isNaN(lat) || isNaN(lng)) return;
  const idx = images.findIndex(i => i.id === img.id) + 1;
  if (uploadPins[img.id]) {
    uploadPins[img.id].setLatLng([lat, lng]);
    uploadPins[img.id].setIcon(makeIcon(idx, '#2563eb'));
  } else {
    uploadPins[img.id] = L.marker([lat, lng], {icon: makeIcon(idx, '#2563eb')})
      .bindPopup(`<b>${img.name}</b><br>${img.area}`)
      .addTo(uploadMap);
  }
}

// ─── Analyze ─────────────────────────────────────────────
async function analyzeAll() {
  if (images.length === 0) return;
  const btn = document.getElementById('analyze-btn');
  btn.disabled = true;
  btn.innerHTML = `<span style="animation:spin 1s linear infinite;display:inline-block">⏳</span> Analyzing ${images.length} image(s)…`;

  const fd = new FormData();
  images.forEach(img => {
    fd.append('images', img.file);
    fd.append('lats', img.lat);
    fd.append('lngs', img.lng);
    fd.append('names', img.area || img.name);
  });

  try {
    const resp = await fetch('/predict-batch', {method:'POST', body:fd});
    const data = await resp.json();
    showResults(data.results);
  } catch(err) {
    alert('Analysis failed: ' + err);
    btn.disabled = false;
    btn.innerHTML = '🔍 Analyze Flood Depths';
  }
}

// ─── Results View ─────────────────────────────────────────
function showResults(results) {
  document.getElementById('v-upload').style.display = 'none';
  const vr = document.getElementById('v-results');
  vr.style.display = 'flex';

  document.getElementById('r-subtitle').textContent =
    `${results.length} location${results.length !== 1 ? 's' : ''} analyzed`;

  // Legend counts
  const counts = {};
  STAGES.forEach(s => counts[s.level] = 0);
  results.forEach(r => { if (r.severity) counts[r.severity.level] = (counts[r.severity.level]||0)+1; });
  document.getElementById('legend').innerHTML = STAGES.map(s => `
    <div class="legend-row">
      <div class="legend-dot" style="background:${s.color}"></div>
      <div class="legend-label"><b>Stage ${STAGES.indexOf(s)+1}</b> ${s.level} — ${s.label} (${s.min}–${s.max === 999 ? '80+' : s.max} cm)</div>
      <div class="legend-count" style="color:${s.color}">${counts[s.level]||0}</div>
    </div>`).join('');

  // Results list
  document.getElementById('results-list').innerHTML = results.map((r, i) => {
    if (r.status === 'error') return `<div class="r-card" style="border-left-color:#ef4444">
      <div class="r-card-name">${r.name}</div>
      <div style="font-size:.8rem;color:#ef4444">Error: ${r.error}</div></div>`;
    const s = r.severity;
    const methodBadge = r.method === 'reference_object_cv'
      ? `<span style="background:#f59e0b;color:#fff;border-radius:4px;padding:1px 6px;font-size:.65rem;font-weight:700">CV FALLBACK</span>`
      : (r.method === 'gemini' ? `<span style="background:#3b82f6;color:#fff;border-radius:4px;padding:1px 6px;font-size:.65rem;font-weight:700">GEMINI</span>` : '');
    const cueHtml = (r.visual_cues && r.visual_cues.length)
      ? `<div style="font-size:.67rem;color:#64748b;margin-top:3px">🔍 ${r.visual_cues.slice(0,2).join(' · ')}</div>` : '';
    const refHtml = (r.reference_objects && r.reference_objects.length)
      ? `<div style="font-size:.67rem;color:#475569;margin-top:2px">📏 ${r.reference_objects.slice(0,3).map(o =>
          `${o.name.replace(/_/g,' ')} (${o.known_height_cm}cm) → ${o.depth_estimate_cm}cm`).join(' · ')}</div>` : '';
    return `<div class="r-card" id="rc-${i}" style="border-left-color:${s.color}" onclick="flyTo(${r.lat},${r.lng},${i})">
      <div class="r-card-name">${r.name} ${methodBadge}</div>
      <div class="r-card-depth" style="color:${s.color}">${r.depth_cm} cm${
        (r.depth_range_cm && r.depth_range_cm[0] !== r.depth_range_cm[1])
          ? ` <span style="font-size:.6em;color:#64748b;font-weight:400">(${r.depth_range_cm[0]}–${r.depth_range_cm[1]})</span>` : ''}</div>
      <div class="r-card-level" style="color:${s.color}">${s.level} — ${s.label}</div>
      ${cueHtml}
      ${refHtml}
      <div class="r-card-loc">📍 ${r.lat.toFixed(4)}, ${r.lng.toFixed(4)}</div>
    </div>`;
  }).join('');

  // Init / reset results map
  if (!resultsMap) {
    resultsMap = L.map('results-map').setView([12.9716, 77.5946], 11);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors', maxZoom: 18
    }).addTo(resultsMap);
  } else {
    resultsMap.eachLayer(l => { if (l instanceof L.Circle || l instanceof L.Marker) resultsMap.removeLayer(l); });
  }
  setTimeout(() => resultsMap.invalidateSize(), 50);

  const bounds = [];
  results.forEach((r, i) => {
    if (r.status === 'error' || !r.severity) return;
    const s = r.severity;
    const radius = 300 + (r.depth_cm / 100) * 700;
    const circle = L.circle([r.lat, r.lng], {
      radius, color: s.color, fillColor: s.color,
      fillOpacity: 0.45, weight: 2, opacity: 0.9
    }).addTo(resultsMap);

    const popup = `<div style="font-family:system-ui;min-width:160px">
      <div style="font-weight:700;font-size:.9rem;margin-bottom:4px">${r.name}</div>
      <div style="font-size:1.4rem;font-weight:800;color:${s.color}">${r.depth_cm} cm</div>
      <div style="color:${s.color};font-size:.8rem;font-weight:600">${s.level} — ${s.label}</div>
      <div style="font-size:.75rem;color:#64748b;margin-top:4px">Confidence: ${(r.confidence*100).toFixed(1)}%</div>
      ${r.visual_cues && r.visual_cues.length ? `<div style="font-size:.7rem;color:#64748b;margin-top:2px">🔍 ${r.visual_cues.slice(0,2).join('<br>')}</div>` : ''}
      <div style="font-size:.72rem;color:#94a3b8">${r.lat.toFixed(5)}, ${r.lng.toFixed(5)}</div>
    </div>`;
    circle.bindPopup(popup);

    const label = L.divIcon({
      className:'',
      html:`<div style="background:${s.color};color:#fff;border-radius:10px;padding:2px 8px;font-size:11px;font-weight:700;white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,.3)">${r.depth_cm}cm</div>`,
      iconAnchor:[0,-8]
    });
    L.marker([r.lat, r.lng], {icon:label, interactive:false}).addTo(resultsMap);

    bounds.push([r.lat, r.lng]);
  });

  if (bounds.length > 1) resultsMap.fitBounds(bounds, {padding:[40,40]});
  else if (bounds.length === 1) resultsMap.setView(bounds[0], 13);
}

function flyTo(lat, lng, idx) {
  resultsMap.flyTo([lat, lng], 14, {duration: 0.8});
}

function showUpload() {
  document.getElementById('v-results').style.display = 'none';
  document.getElementById('v-upload').style.display = 'flex';
  setTimeout(() => uploadMap.invalidateSize(), 50);
  document.getElementById('analyze-btn').disabled = images.length === 0;
  document.getElementById('analyze-btn').innerHTML = '🔍 Analyze Flood Depths';
}

// spin keyframe
const st = document.createElement('style');
st.textContent = '@keyframes spin{to{transform:rotate(360deg)}}';
document.head.appendChild(st);
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────
# Flask app
# ─────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB for batch


@app.get("/")
def index():
    return render_template_string(HTML, areas=json.dumps(BENGALURU_AREAS))


def _predict_single(image: Image.Image) -> dict:
    """
    Run depth prediction on a PIL Image.
    Primary path: Gemini vision endpoint. Falls back to the reference-object
    CV estimator when Gemini is unconfigured or a call fails.
    """
    result = None
    method = "reference_object_cv"
    if _GEMINI_ESTIMATOR.available:
        try:
            result = _GEMINI_ESTIMATOR.estimate(image)
            method = "gemini"
        except GeminiError as exc:
            logger.warning(f"Gemini prediction failed, falling back to reference CV: {exc}")

    if result is None:
        img_arr = np.array(image)
        result = _REFERENCE_ESTIMATOR.estimate(img_arr)

    depth_cm = result["depth_cm"]
    confidence = result["confidence"]
    visual_cues = result.get("visual_cues", [])
    label_guide = result.get("label_guide", "")
    waterline_pct = result.get("waterline_pct", 0)
    water_coverage = result.get("water_coverage", 0)
    reference_objects = result.get("reference_objects", [])

    depth_cm = round(depth_cm, 2)
    return {
        "depth_cm": depth_cm,
        "depth_range_cm": result.get("depth_range_cm") or [depth_cm, depth_cm],
        "confidence": round(confidence, 4),
        "severity": depth_to_severity(depth_cm),
        "method": method,
        "visual_cues": visual_cues,
        "label_guide": label_guide,
        "scene_analysis": result.get("scene_analysis", ""),
        "waterline_pct": waterline_pct,
        "water_coverage": water_coverage,
        "reference_objects": reference_objects,
    }


@app.post("/predict-batch")
def predict_batch():
    files = request.files.getlist("images")
    lats = request.form.getlist("lats")
    lngs = request.form.getlist("lngs")
    names = request.form.getlist("names")

    results = []
    for i, file in enumerate(files):
        lat = float(lats[i]) if i < len(lats) and lats[i] else 12.9716
        lng = float(lngs[i]) if i < len(lngs) and lngs[i] else 77.5946
        name = (names[i] if i < len(names) and names[i] else file.filename) or f"Image {i+1}"

        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            pred = _predict_single(image)
            logger.info(f"  [{i+1}] {name}: {pred['depth_cm']} cm ({pred['severity']['level']}) [{pred['method']}]")
            results.append({"name": name, "lat": lat, "lng": lng, "status": "ok", **pred})
        except Exception as e:
            logger.error(f"  [{i+1}] {name}: error — {e}")
            results.append({"name": name, "lat": lat, "lng": lng, "error": str(e), "status": "error"})

    return jsonify({"results": results, "count": len(results)})


@app.post("/predict")
def predict():
    """Single-image prediction (kept for compatibility)."""
    if "image" not in request.files:
        return jsonify({"error": "No image field"}), 400
    file = request.files["image"]
    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    pred = _predict_single(image)
    return jsonify({**pred, "backend": "gemini" if _GEMINI_ESTIMATOR.available else "reference_object_cv"})


@app.get("/health")
def health():
    gemini_ok = _GEMINI_ESTIMATOR.available
    return jsonify({
        "status": "ok",
        "model": _GEMINI_ESTIMATOR.model_name,
        "gemini_available": gemini_ok,
        "warning": None if gemini_ok else "gemini_api_key_missing",
        "active_method": "gemini" if gemini_ok else "reference_object_cv",
        "reference_cv_available": True,
    })



@app.post("/ingest")
def ingest():
    """
    Unified ingestion endpoint using shared event contract.
    Same payload contract is used for API and queue execution paths.
    """
    from pydantic import ValidationError
    from src.event_contract import FloodEvent
    from src.middleware.retry import RetryPolicy
    from src.pipeline import execute_event

    raw = request.get_json(force=True, silent=True)
    if not raw:
        return jsonify({"error": "JSON body required"}), 400

    try:
        event_payload = {
            "source": "api",
            "camera_id": raw.get("camera_id", ""),
            "latitude": raw.get("latitude"),
            "longitude": raw.get("longitude"),
            "image_b64": raw.get("image", ""),
            "metadata": {"client_ip": request.remote_addr},
        }
        if raw.get("timestamp"):
            event_payload["timestamp"] = raw.get("timestamp")
        if request.headers.get("X-Trace-Id"):
            event_payload["trace_id"] = request.headers.get("X-Trace-Id")
        payload = FloodEvent(**event_payload)
    except ValidationError as exc:
        return jsonify({"error": "Validation failed", "detail": exc.errors()}), 422

    try:
        from tasks import infer_flood_depth
        task = infer_flood_depth.apply_async(args=[payload.to_task_payload()])
        return jsonify({
            "task_id": task.id,
            "status": "queued",
            "camera_id": payload.camera_id,
            "event_id": payload.event_id,
            "trace_id": payload.trace_id,
            "schema_version": payload.schema_version,
        }), 202
    except Exception:
        from src.dlq import get_dead_letter_router
        from src.event_contract import FloodFailureEvent

        try:
            result = execute_event(
                payload,
                retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0.5, max_delay_seconds=6.0),
            )
            return jsonify(result.to_api_response())
        except Exception as exc:
            failure = FloodFailureEvent.from_exception(
                exc=exc,
                stage="api.ingest.sync_fallback",
                attempts=3,
                max_attempts=3,
                retry_exhausted=True,
                event=payload,
                source="api",
                metadata={"adapter": "app.ingest"},
            )
            dlq_info = get_dead_letter_router().publish(failure)
            failure.metadata["dlq"] = dlq_info
            return jsonify(failure.to_api_response()), 500


@app.post("/export/geojson")
def export_geojson():
    """
    Phase 5: Export prediction list as GeoJSON FeatureCollection.
    Body: {"predictions": [...list of prediction dicts...]}
    """
    from src.geospatial_classifier import FloodIntensityClassifier
    raw = request.get_json(force=True, silent=True) or {}
    predictions = raw.get("predictions", [])
    if not predictions:
        return jsonify({"error": "predictions array required"}), 400
    clf = FloodIntensityClassifier()
    fc = clf.to_geojson(predictions)
    return jsonify(fc)


if __name__ == "__main__":
    logger.info("🌊 Starting Flood Depth Estimator at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
