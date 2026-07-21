# ui/upload_app.py
"""
FloodWatch AI — Upload & Predict Web App
=========================================
Self-contained Streamlit app for single-image testing and demos.
No ingestor / batch queue needed — upload → predict → show map instantly.

Run: streamlit run ui/upload_app.py

Features:
  • Upload one or many images with camera metadata
  • Runs full 5-stage pipeline + optional Gemini ensemble
  • Shows Bengaluru map with 5 outputs per camera
  • Explainability panel: model vs Gemini comparison
  • Export CSV report
"""

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

# ── Try loading real pipeline; fall back to direct predict.py ────────────────
try:
    from pipeline.runner  import PipelineRunner
    from ingestor         import CameraImage
    _HAS_PIPELINE = True
except ImportError:
    _HAS_PIPELINE = False

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloodWatch AI — Upload & Predict",
    page_icon="🌊", layout="wide",
)

RISK_STYLE = {
    "NO FLOOD":  {"bg": "#2ecc71", "icon": "✅", "fc": "green"},
    "LOW RISK":  {"bg": "#3498db", "icon": "🟦", "fc": "blue"},
    "MODERATE":  {"bg": "#f39c12", "icon": "🟠", "fc": "orange"},
    "HIGH RISK": {"bg": "#e74c3c", "icon": "🔴", "fc": "red"},
    "CRITICAL":  {"bg": "#7b241c", "icon": "🚨", "fc": "darkred"},
}

CAMERAS = {
    "Silk Board Junction":               {"lat": 12.9172, "lon": 77.6228},
    "Bellandur Lake Road":               {"lat": 12.9254, "lon": 77.6784},
    "Marathahalli Bridge":               {"lat": 12.9563, "lon": 77.7010},
    "Hebbal Flyover":                    {"lat": 13.0351, "lon": 77.5975},
    "Ulsoor Road":                       {"lat": 12.9784, "lon": 77.6206},
    "Whitefield Main Road":              {"lat": 12.9698, "lon": 77.7500},
    "Koramangala 5th Block":             {"lat": 12.9352, "lon": 77.6245},
    "Custom ▸ enter manually":           {"lat": 12.9716, "lon": 77.5946},
}

# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = []   # list of result dicts

# ── Cached pipeline ───────────────────────────────────────────────────────────
@st.cache_resource
def get_pipeline(gemini_key: str = ""):
    cfg = {
        "pipeline": {
            "device":          "cpu",
            "gemini_api_key":  gemini_key or os.environ.get("GEMINI_API_KEY", ""),
        }
    }
    if _HAS_PIPELINE:
        return PipelineRunner(cfg)
    return None


def run_on_image(img_bgr, camera_id, location_id, lat, lon, loc_name, gemini_key):
    pipeline = get_pipeline(gemini_key)
    if pipeline is None:
        return _heuristic_fallback(img_bgr, camera_id, location_id, lat, lon, loc_name)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, img_bgr)
        tmp_path = Path(f.name)

    cam_img = CameraImage(
        image_path    = tmp_path,
        camera_id     = camera_id,
        location_id   = location_id,
        latitude      = lat,
        longitude     = lon,
        location_name = loc_name,
        captured_at   = datetime.now().isoformat(),
    )
    pred = pipeline.run_image(cam_img, batch_id="upload")
    tmp_path.unlink(missing_ok=True)
    return pred.__dict__


def _heuristic_fallback(img_bgr, camera_id, location_id, lat, lon, loc_name):
    """Used when pipeline package is not available."""
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = img_bgr[int(img_bgr.shape[0]*0.5):, :]
    hsv_l = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    blue  = cv2.inRange(hsv_l, np.array([85,20,30]),  np.array([135,255,255]))
    dark  = cv2.inRange(hsv_l, np.array([0,0,20]),    np.array([180,60,150]))
    pct   = np.count_nonzero(cv2.bitwise_or(blue,dark)) / (lower.shape[0]*lower.shape[1]) * 100
    depth = min(round(pct * 1.8, 1), 120.0)
    risk  = (["NO FLOOD","LOW RISK","MODERATE","HIGH RISK","CRITICAL"]
             [min(4, int(depth//15))])
    return {"camera_id":camera_id,"location_id":location_id,"latitude":lat,"longitude":lon,
            "location_name":loc_name,"water_depth_cm":depth,"risk_level":risk,
            "flood_detected":depth>0,"confidence_pct":55.0,
            "recommended_action":"","ensemble_method":"heuristic",
            "gemini_risk":None,"gemini_depth_cm":None,"gemini_reasoning":None,
            "gemini_agreement":None,"timestamp":datetime.now().isoformat()}


def build_map(results):
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB positron")
    for r in results:
        risk = r.get("risk_level","NO FLOOD")
        sty  = RISK_STYLE.get(risk, RISK_STYLE["NO FLOOD"])
        d    = r.get("water_depth_cm",0)
        popup_html = f"""
        <div style="font-family:Arial;min-width:250px">
          <h4 style="margin:0;color:{sty['bg']}">{sty['icon']} {risk}</h4><hr>
          <b>① Flood:</b> {"YES 🌊" if r.get("flood_detected") else "NO ✅"}<br>
          <b>② Depth:</b> {d} cm<br>
          <b>③ Risk:</b> <span style="color:{sty['bg']}">{risk}</span><br>
          <b>④ Action:</b> {r.get("recommended_action","")}<br>
          <b>⑤ Confidence:</b> {r.get("confidence_pct",0)}%<br><hr>
          <small>Method: {r.get("ensemble_method","")}</small>
        </div>"""
        folium.Marker(
            location=[r["latitude"], r["longitude"]],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{sty['icon']} {r.get('location_name','')} — {d}cm",
            icon=folium.Icon(color=sty["fc"], icon="tint", prefix="fa"),
        ).add_to(m)
        if risk in ("CRITICAL","HIGH RISK"):
            folium.CircleMarker([r["latitude"],r["longitude"]],
                                radius=28, color=sty["bg"], fill=False, weight=2, opacity=0.5
                                ).add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌊 FloodWatch AI\nUpload & Predict")
    st.divider()

    st.subheader("📥 Camera Metadata")
    location_id  = st.text_input("Location ID",  value="LOC_001")
    camera_id    = st.text_input("Camera ID",    value="CAM_001")
    cam_sel      = st.selectbox("Location",      list(CAMERAS.keys()))

    if cam_sel == "Custom ▸ enter manually":
        lat      = st.number_input("Latitude",  value=12.9716, format="%.6f")
        lon      = st.number_input("Longitude", value=77.5946, format="%.6f")
        loc_name = st.text_input("Location Name", value="Custom")
    else:
        lat      = CAMERAS[cam_sel]["lat"]
        lon      = CAMERAS[cam_sel]["lon"]
        loc_name = cam_sel

    st.divider()
    st.subheader("🤖 Optional: Gemini Ensemble")
    gemini_key = st.text_input("Gemini API Key", type="password",
                                value=os.environ.get("GEMINI_API_KEY",""),
                                help="Leave blank to use model-only prediction")
    if gemini_key:
        st.success("Gemini ensemble enabled ✅")
    else:
        st.caption("Model-only mode (no Gemini)")

    st.divider()
    uploaded_files = st.file_uploader(
        "📷 Upload Image(s)", type=["jpg","jpeg","png"],
        accept_multiple_files=True,
    )
    run_btn = st.button("🔍 Run Prediction", type="primary", use_container_width=True)
    st.divider()
    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.results = []
        st.rerun()

    st.subheader("Legend")
    for r, s in RISK_STYLE.items():
        st.markdown(f"{s['icon']} {r}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🌊 FloodWatch AI — Upload & Predict")
st.caption("Upload camera images → runs 5-stage pipeline + optional Gemini ensemble → Bengaluru map")
st.divider()

# KPI bar
rds = st.session_state.results
c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("📷 Processed",   len(rds))
c2.metric("⚠️ Floods",       sum(1 for r in rds if r.get("flood_detected")), delta_color="inverse")
c3.metric("🚨 Critical",     sum(1 for r in rds if r.get("risk_level")=="CRITICAL"), delta_color="inverse")
c4.metric("💧 Max Depth",    f"{max((r.get('water_depth_cm',0) for r in rds), default=0)} cm")
c5.metric("🤖 Gemini",       "ON ✅" if gemini_key else "OFF")
st.divider()

# ── Run predictions ───────────────────────────────────────────────────────────
if uploaded_files and run_btn:
    progress = st.progress(0, text="Running pipeline…")
    for i, uf in enumerate(uploaded_files):
        img_bytes = np.frombuffer(uf.read(), np.uint8)
        img_bgr   = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        cam_id    = f"{camera_id}_{i+1:02d}" if len(uploaded_files) > 1 else camera_id

        with st.spinner(f"Processing {uf.name}…"):
            result = run_on_image(img_bgr, cam_id, location_id, lat, lon, loc_name, gemini_key)
            result["filename"] = uf.name

        st.session_state.results = [r for r in st.session_state.results
                                     if r.get("camera_id") != cam_id]
        st.session_state.results.append(result)
        progress.progress((i+1)/len(uploaded_files), f"Done {i+1}/{len(uploaded_files)}")
    st.rerun()

# ── Map + results ─────────────────────────────────────────────────────────────
map_col, res_col = st.columns([3,2])

with map_col:
    st.subheader("🗺️ Bengaluru Flood Map")
    st_folium(build_map(st.session_state.results), width=700, height=480, returned_objects=[])

with res_col:
    st.subheader("📤 Results")
    for r in reversed(st.session_state.results):
        sty = RISK_STYLE.get(r.get("risk_level","NO FLOOD"), RISK_STYLE["NO FLOOD"])
        with st.expander(f"{sty['icon']} {r.get('location_name','')} — {r.get('water_depth_cm',0)} cm"):
            a,b = st.columns(2)
            a.metric("① Flood",     "YES 🌊" if r.get("flood_detected") else "NO ✅")
            b.metric("② Depth",     f"{r.get('water_depth_cm',0)} cm")
            a.metric("③ Risk",      r.get("risk_level",""))
            b.metric("⑤ Confidence",f"{r.get('confidence_pct',0)}%")
            st.info(f"**④ Action:** {r.get('recommended_action','')}")

            # Gemini comparison
            if r.get("gemini_risk"):
                st.markdown("---\n**🤖 Gemini Ensemble**")
                c1,c2 = st.columns(2)
                c1.metric("Model",  r.get("risk_level",""), f"{r.get('water_depth_cm',0)}cm")
                c2.metric("Gemini", r.get("gemini_risk",""), f"{r.get('gemini_depth_cm',0)}cm")
                agree = r.get("gemini_agreement")
                st.markdown(f"{'✅ Agreed' if agree else '⚠️ Disagreed'} — "
                            f"_{r.get('gemini_reasoning','')}_")

# ── Full table ────────────────────────────────────────────────────────────────
st.divider()
if st.session_state.results:
    import pandas as pd
    df = pd.DataFrame(st.session_state.results)
    cols = ["camera_id","location_name","water_depth_cm","risk_level",
            "confidence_pct","ensemble_method","timestamp"]
    df2 = df[[c for c in cols if c in df.columns]]
    st.dataframe(df2, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Export CSV", df2.to_csv(index=False),
                       f"flood_{datetime.now().strftime('%Y%m%d_%H%M')}.csv","text/csv")

st.divider()
st.caption("FloodWatch AI | SegFormer + YOLOv8 + Depth Anything V2 + Gemini Ensemble")
