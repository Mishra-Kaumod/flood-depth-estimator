# ui/upload_app.py
"""
FloodWatch AI — Upload & Predict
================================
Upload one or more images.  For each image you can optionally supply
latitude, longitude, and a camera ID.  Any blank field is auto-filled
with a randomly chosen BBMP Bengaluru camera location.

Run:  streamlit run ui/upload_app.py
"""

import os
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
    _HAS_FOLIUM = True
except ImportError:
    _HAS_FOLIUM = False

try:
    from pipeline.runner import PipelineRunner
    from ingestor        import CameraImage
    _HAS_PIPELINE = True
except ImportError:
    _HAS_PIPELINE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloodWatch AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #f8f9fa;
    font-family: 'Inter', 'Google Sans', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: #fff;
    border-right: 1px solid #dadce0;
}
[data-testid="stSidebar"] label {
    font-size: 0.79rem !important;
    color: #3c4043 !important;
    font-weight: 500 !important;
}

/* ── Header ── */
.fw-header {
    background: linear-gradient(135deg, #1a73e8 0%, #1557b0 100%);
    color: #fff;
    padding: 20px 28px 16px;
    border-radius: 12px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 2px 8px rgba(26,115,232,.25);
}
.fw-header .fw-logo { font-size: 2.4rem; line-height: 1; }
.fw-header h1  { margin: 0; font-size: 1.55rem; font-weight: 700; letter-spacing: -.01em; }
.fw-header p   { margin: 3px 0 0; font-size: 0.82rem; opacity: .82; }
.fw-header .fw-badge {
    margin-left: auto;
    background: rgba(255,255,255,.18);
    border: 1px solid rgba(255,255,255,.35);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 600;
    white-space: nowrap;
}

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 10px; margin-bottom: 18px; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 110px;
    background: #fff;
    border: 1px solid #dadce0;
    border-radius: 10px;
    padding: 14px 14px 10px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.kpi-card .kv  { font-size: 1.9rem; font-weight: 700; color: #1a73e8; line-height: 1; }
.kpi-card .kl  { font-size: 0.68rem; color: #5f6368; margin-top: 4px; text-transform: uppercase; letter-spacing: .05em; }
.kpi-card.warn .kv { color: #e37400; }
.kpi-card.crit .kv { color: #d93025; }
.kpi-card.good .kv { color: #1e8e3e; }

/* ── Metadata editor heading ── */
.meta-section {
    background: #fff;
    border: 1px solid #dadce0;
    border-radius: 10px;
    padding: 14px 16px 10px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.meta-section .ms-title {
    font-size: 0.8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .06em;
    color: #5f6368; margin-bottom: 10px;
}

/* ── Result cards ── */
.result-card {
    background: #fff;
    border: 1px solid #dadce0;
    border-left: 5px solid #1a73e8;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
.result-card.rc-crit { border-left-color: #d93025; }
.result-card.rc-high { border-left-color: #e37400; }
.result-card.rc-mod  { border-left-color: #f9ab00; }
.result-card .rct { font-size: 0.88rem; font-weight: 700; color: #202124; margin-bottom: 8px; }
.result-card .rcr { display:flex; gap:14px; flex-wrap:wrap; }
.result-card .rcs { flex:1; min-width:72px; }
.result-card .rcn { font-size:1.25rem; font-weight:700; color:#1a73e8; }
.result-card .rcl { font-size:0.68rem; color:#80868b; text-transform:uppercase; letter-spacing:.04em; }
.result-card .rca {
    margin-top:8px; font-size:0.78rem; color:#3c4043;
    background:#f1f3f4; border-radius:6px; padding:6px 10px;
}

/* ── Gemini strip ── */
.gemini-strip {
    background:#e8f0fe; border:1px solid #c5d5f5; border-radius:8px;
    padding:8px 12px; margin-top:-4px; margin-bottom:10px; font-size:0.79rem; color:#1558b0;
}

/* ── Section label ── */
.sec-lbl {
    font-size: 0.73rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.07em; color:#5f6368; margin:14px 0 7px;
}

/* ── Sidebar brand ── */
.sb-brand { padding:10px 0 2px; display:flex; align-items:center; gap:10px; }
.sb-brand .sb-logo { font-size:1.7rem; }
.sb-brand .sb-name { font-weight:700; font-size:0.97rem; color:#202124; }
.sb-brand .sb-ver  { font-size:0.69rem; color:#5f6368; }

[data-testid="stDataFrame"] { border-radius:8px; overflow:hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_STYLE = {
    "NO FLOOD":  {"bg":"#1e8e3e","light":"#e6f4ea","icon":"✅","fc":"green",   "rc":""},
    "LOW RISK":  {"bg":"#1a73e8","light":"#e8f0fe","icon":"💧","fc":"blue",    "rc":""},
    "MODERATE":  {"bg":"#f9ab00","light":"#fef7e0","icon":"🟠","fc":"orange",  "rc":"rc-mod"},
    "HIGH RISK": {"bg":"#e37400","light":"#fce8e6","icon":"🔴","fc":"red",     "rc":"rc-high"},
    "CRITICAL":  {"bg":"#d93025","light":"#fce8e6","icon":"🚨","fc":"darkred", "rc":"rc-crit"},
}

# Known BBMP Bengaluru camera locations
CAMERAS = {
    "Silk Board Junction":     {"lat":12.9172,"lon":77.6228,"cam":"CAM_001"},
    "Bellandur Lake Road":     {"lat":12.9254,"lon":77.6784,"cam":"CAM_002"},
    "Marathahalli Bridge":     {"lat":12.9563,"lon":77.7010,"cam":"CAM_003"},
    "Hebbal Flyover":          {"lat":13.0351,"lon":77.5975,"cam":"CAM_004"},
    "Ulsoor Road":             {"lat":12.9784,"lon":77.6206,"cam":"CAM_005"},
    "Whitefield Main Road":    {"lat":12.9698,"lon":77.7500,"cam":"CAM_006"},
    "Koramangala 5th Block":   {"lat":12.9352,"lon":77.6245,"cam":"CAM_007"},
    "K R Puram Bridge":        {"lat":13.0005,"lon":77.6961,"cam":"CAM_008"},
    "Nagawara Junction":       {"lat":13.0400,"lon":77.6270,"cam":"CAM_009"},
    "Tin Factory":             {"lat":12.9957,"lon":77.6598,"cam":"CAM_010"},
}
CAMERA_NAMES = list(CAMERAS.keys())

# ── Session state ─────────────────────────────────────────────────────────────
if "results"  not in st.session_state: st.session_state.results  = []
if "img_meta" not in st.session_state: st.session_state.img_meta = {}  # fname -> meta dict


# ── Helpers ───────────────────────────────────────────────────────────────────
def random_camera() -> dict:
    """Pick a random BBMP camera and return its metadata."""
    name = random.choice(CAMERA_NAMES)
    c    = CAMERAS[name]
    return {"location_name": name, "lat": c["lat"], "lon": c["lon"], "camera_id": c["cam"]}


def build_meta_df(uploaded_files) -> pd.DataFrame:
    """
    Build the per-image metadata DataFrame.
    Existing session-state values are preserved; new files get random cameras.
    """
    rows = []
    for uf in uploaded_files:
        saved = st.session_state.img_meta.get(uf.name)
        if saved:
            rows.append(saved)
        else:
            rc = random_camera()
            rows.append({
                "filename":      uf.name,
                "camera_id":     rc["camera_id"],
                "location_name": rc["location_name"],
                "latitude":      rc["lat"],
                "longitude":     rc["lon"],
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["filename","camera_id","location_name","latitude","longitude"]
    )


def apply_random_to_blanks(df: pd.DataFrame) -> pd.DataFrame:
    """Fill any blank/NaN metadata cells with random camera values."""
    df = df.copy()
    for i, row in df.iterrows():
        need_fill = (
            not str(row.get("camera_id","")).strip()
            or pd.isna(row.get("latitude"))
            or pd.isna(row.get("longitude"))
            or not str(row.get("location_name","")).strip()
        )
        if need_fill:
            rc = random_camera()
            if not str(row.get("camera_id","")).strip():
                df.at[i,"camera_id"] = rc["camera_id"]
            if pd.isna(row.get("latitude")) or row.get("latitude","") == "":
                df.at[i,"latitude"]  = rc["lat"]
            if pd.isna(row.get("longitude")) or row.get("longitude","") == "":
                df.at[i,"longitude"] = rc["lon"]
            if not str(row.get("location_name","")).strip():
                df.at[i,"location_name"] = rc["location_name"]
    return df


# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_pipeline(gemini_key: str = ""):
    cfg = {"pipeline": {"device": "cpu",
                        "gemini_api_key": gemini_key or os.environ.get("GEMINI_API_KEY","")}}
    return PipelineRunner(cfg) if _HAS_PIPELINE else None


def run_on_image(img_bgr, camera_id, location_id, lat, lon, loc_name, gemini_key):
    pipeline = get_pipeline(gemini_key)
    if pipeline is None:
        return _heuristic_fallback(img_bgr, camera_id, location_id, lat, lon, loc_name)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, img_bgr)
        tmp = Path(f.name)
    cam_img = CameraImage(image_path=tmp, camera_id=camera_id,
                          location_id=location_id, latitude=lat, longitude=lon,
                          location_name=loc_name,
                          captured_at=datetime.now().isoformat())
    pred = pipeline.run_image(cam_img, batch_id="upload")
    tmp.unlink(missing_ok=True)
    return pred.__dict__


def _heuristic_fallback(img_bgr, camera_id, location_id, lat, lon, loc_name):
    lower = img_bgr[int(img_bgr.shape[0]*0.5):, :]
    hsv   = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    blue  = cv2.inRange(hsv, np.array([85,20,30]),  np.array([135,255,255]))
    dark  = cv2.inRange(hsv, np.array([0,0,20]),    np.array([180,60,150]))
    pct   = np.count_nonzero(cv2.bitwise_or(blue,dark)) / (lower.shape[0]*lower.shape[1]) * 100
    depth = min(round(pct*1.8,1), 120.0)
    risk  = ["NO FLOOD","LOW RISK","MODERATE","HIGH RISK","CRITICAL"][min(4,int(depth//15))]
    return {"camera_id":camera_id,"location_id":location_id,"latitude":lat,"longitude":lon,
            "location_name":loc_name,"water_depth_cm":depth,"risk_level":risk,
            "flood_detected":depth>0,"confidence_pct":55.0,
            "recommended_action":"Monitor situation","ensemble_method":"heuristic",
            "gemini_risk":None,"gemini_depth_cm":None,"gemini_reasoning":None,
            "gemini_agreement":None,"timestamp":datetime.now().isoformat()}


# ── Map with Google Maps tiles ────────────────────────────────────────────────
def build_map(results: list) -> "folium.Map":
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12)

    # Google Maps roadmap tile layer
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="© Google Maps",
        name="Google Maps",
        max_zoom=20,
    ).add_to(m)
    # Google Satellite hybrid option
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="© Google Satellite",
        name="Google Satellite",
        max_zoom=20,
    ).add_to(m)
    folium.LayerControl(position="topright").add_to(m)

    for r in results:
        risk = r.get("risk_level","NO FLOOD")
        sty  = RISK_STYLE.get(risk, RISK_STYLE["NO FLOOD"])
        d    = r.get("water_depth_cm",0)
        conf = r.get("confidence_pct",0)
        popup_html = f"""
        <div style="font-family:'Inter','Google Sans',sans-serif;min-width:230px;padding:4px">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span style="font-size:1.15rem">{sty['icon']}</span>
            <strong style="color:{sty['bg']};font-size:1rem">{risk}</strong>
          </div>
          <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
            <tr><td style="color:#5f6368;padding:2px 4px 2px 0">📷 Camera</td>
                <td style="font-weight:600">{r.get('camera_id','')}</td></tr>
            <tr><td style="color:#5f6368;padding:2px 4px 2px 0">📍 Location</td>
                <td style="font-weight:600">{r.get('location_name','')}</td></tr>
            <tr><td style="color:#5f6368;padding:2px 4px 2px 0">💧 Depth</td>
                <td style="font-weight:600">{d} cm</td></tr>
            <tr><td style="color:#5f6368;padding:2px 4px 2px 0">🎯 Confidence</td>
                <td style="font-weight:600">{conf:.0f}%</td></tr>
            <tr><td style="color:#5f6368;padding:2px 4px 2px 0">🌊 Flood</td>
                <td style="font-weight:600">{"Yes" if r.get("flood_detected") else "No"}</td></tr>
          </table>
          <div style="margin-top:8px;background:{sty['bg']};color:#fff;border-radius:6px;
                      padding:5px 8px;font-size:0.78rem;font-weight:600">
            ⚡ {r.get('recommended_action','Monitor')}
          </div>
          <div style="margin-top:5px;font-size:0.7rem;color:#80868b">
            {r.get('ensemble_method','model')} · {r.get('filename','')}<br>
            {r.get('timestamp','')[:16]}
          </div>
        </div>"""
        folium.Marker(
            location=[r["latitude"], r["longitude"]],
            popup=folium.Popup(popup_html, max_width=270),
            tooltip=f"{sty['icon']} {r.get('location_name','')} — {d} cm",
            icon=folium.Icon(color=sty["fc"], icon="tint", prefix="fa"),
        ).add_to(m)
        if risk in ("CRITICAL","HIGH RISK"):
            folium.CircleMarker(
                [r["latitude"],r["longitude"]],
                radius=30, color=sty["bg"], fill=True,
                fill_color=sty["bg"], fill_opacity=0.08,
                weight=2, opacity=0.45,
            ).add_to(m)
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <span class="sb-logo">🌊</span>
      <div>
        <div class="sb-name">FloodWatch AI</div>
        <div class="sb-ver">BBMP Bengaluru · v3.0</div>
      </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="sec-lbl">📷 Upload Images</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drag & drop or browse",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sec-lbl">⚙️ Settings</div>', unsafe_allow_html=True)
    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY",""),
        help="Optional — enables ensemble validation",
        placeholder="AIza...",
    )
    if gemini_key:
        st.success("Gemini ensemble ON ✅", icon="🤖")

    st.markdown("")
    run_btn = st.button("▶  Run Prediction", type="primary", use_container_width=True)
    if st.button("🗑️  Clear results", use_container_width=True):
        st.session_state.results  = []
        st.session_state.img_meta = {}
        st.rerun()

    st.divider()
    st.markdown('<div class="sec-lbl">Risk Legend</div>', unsafe_allow_html=True)
    for rk, sty in RISK_STYLE.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
            f'<span style="width:10px;height:10px;border-radius:50%;display:inline-block;'
            f'background:{sty["bg"]}"></span>'
            f'<span style="font-size:0.79rem;color:#3c4043">{rk}</span></div>',
            unsafe_allow_html=True,
        )
    st.divider()
    st.caption("SegFormer · YOLOv8 · Depth Anything V2 · Gemini")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
gemini_key = gemini_key if "gemini_key" in dir() else ""

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="fw-header">
  <span class="fw-logo">🌊</span>
  <div>
    <h1>FloodWatch AI</h1>
    <p>Upload images · AI 5-stage pipeline · Real-time Bengaluru flood map</p>
  </div>
  <div class="fw-badge">{'🤖 Gemini ON' if gemini_key else '🔵 Model only'}</div>
</div>""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
rds        = st.session_state.results
n_floods   = sum(1 for r in rds if r.get("flood_detected"))
n_critical = sum(1 for r in rds if r.get("risk_level") == "CRITICAL")
max_depth  = max((r.get("water_depth_cm",0) for r in rds), default=0)
avg_conf   = (sum(r.get("confidence_pct",0) for r in rds)/len(rds)) if rds else 0

kpi_floods_cls  = "warn" if n_floods   > 0 else "good"
kpi_crit_cls    = "crit" if n_critical > 0 else ""
kpi_depth_cls   = "warn" if max_depth  > 30 else ""

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kv">{len(rds)}</div>
    <div class="kl">Images processed</div>
  </div>
  <div class="kpi-card {kpi_floods_cls}">
    <div class="kv">{n_floods}</div>
    <div class="kl">Floods detected</div>
  </div>
  <div class="kpi-card {kpi_crit_cls}">
    <div class="kv">{n_critical}</div>
    <div class="kl">Critical zones</div>
  </div>
  <div class="kpi-card {kpi_depth_cls}">
    <div class="kv">{max_depth:.0f} cm</div>
    <div class="kl">Max depth</div>
  </div>
  <div class="kpi-card">
    <div class="kv">{avg_conf:.0f}%</div>
    <div class="kl">Avg confidence</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Per-image metadata editor ─────────────────────────────────────────────────
if uploaded_files:
    st.markdown("""
    <div class="meta-section">
      <div class="ms-title">📋 Image Metadata — Camera ID · Latitude · Longitude</div>
      <div style="font-size:0.77rem;color:#5f6368;margin-bottom:10px">
        Each image has been assigned a random BBMP Bengaluru camera.
        Edit any cell to override — blank fields stay random on run.
      </div>
    </div>""", unsafe_allow_html=True)

    meta_df = build_meta_df(uploaded_files)
    edited  = st.data_editor(
        meta_df,
        column_config={
            "filename":      st.column_config.TextColumn("Image file", disabled=True, width="medium"),
            "camera_id":     st.column_config.TextColumn("Camera ID",  width="small"),
            "location_name": st.column_config.SelectboxColumn(
                "Location", options=CAMERA_NAMES + ["Custom"], width="large",
            ),
            "latitude":      st.column_config.NumberColumn("Latitude",  format="%.4f", min_value=-90,  max_value=90),
            "longitude":     st.column_config.NumberColumn("Longitude", format="%.4f", min_value=-180, max_value=180),
        },
        use_container_width=True,
        hide_index=True,
        key="meta_editor",
    )
    # Persist edits back to session state
    for _, row in edited.iterrows():
        st.session_state.img_meta[row["filename"]] = row.to_dict()

# ── Run predictions ───────────────────────────────────────────────────────────
if uploaded_files and run_btn:
    final_meta = apply_random_to_blanks(edited if uploaded_files else pd.DataFrame())
    meta_by_file = {row["filename"]: row for _, row in final_meta.iterrows()}

    prog = st.progress(0, text="Initialising pipeline…")
    file_map = {uf.name: uf for uf in uploaded_files}

    for i, (fname, row) in enumerate(meta_by_file.items()):
        uf      = file_map.get(fname)
        if uf is None:
            continue
        img_bgr = cv2.imdecode(np.frombuffer(uf.read(), np.uint8), cv2.IMREAD_COLOR)
        cam_id  = str(row.get("camera_id","")).strip() or random_camera()["camera_id"]
        lat     = float(row.get("latitude",  12.9716))
        lon     = float(row.get("longitude", 77.5946))
        loc     = str(row.get("location_name","")).strip() or random_camera()["location_name"]
        loc_id  = f"LOC_{cam_id}"

        with st.spinner(f"Analysing {fname}…"):
            result = run_on_image(img_bgr, cam_id, loc_id, lat, lon, loc, gemini_key)
            result["filename"] = fname

        # Replace any prior result for this camera
        st.session_state.results = [
            r for r in st.session_state.results if r.get("camera_id") != cam_id
        ]
        st.session_state.results.append(result)
        prog.progress((i+1)/len(meta_by_file), f"Done {i+1} / {len(meta_by_file)}")
    st.rerun()

elif run_btn and not uploaded_files:
    st.warning("Please upload at least one image first.", icon="⚠️")

# ── Map + Results ─────────────────────────────────────────────────────────────
map_col, res_col = st.columns([3, 2], gap="medium")

with map_col:
    st.markdown('<div class="sec-lbl">🗺️ Bengaluru Flood Map — Google Maps</div>', unsafe_allow_html=True)
    if _HAS_FOLIUM:
        st_folium(
            build_map(st.session_state.results),
            use_container_width=True,
            height=480,
            returned_objects=[],
        )
    else:
        st.info("Install streamlit-folium for the map: pip install streamlit-folium folium")

with res_col:
    st.markdown('<div class="sec-lbl">📋 Prediction Results</div>', unsafe_allow_html=True)
    if not st.session_state.results:
        st.markdown("""
        <div style="text-align:center;padding:60px 16px;color:#9aa0a6">
          <div style="font-size:3rem">📷</div>
          <div style="margin-top:10px;font-size:0.85rem">
            Upload images and click <strong>Run Prediction</strong>
          </div>
          <div style="margin-top:6px;font-size:0.77rem">
            Lat / lon / camera ID are pre-filled with random<br>BBMP cameras — edit any cell to override.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        for r in reversed(st.session_state.results):
            risk = r.get("risk_level","NO FLOOD")
            sty  = RISK_STYLE.get(risk, RISK_STYLE["NO FLOOD"])
            d    = r.get("water_depth_cm",0)
            conf = r.get("confidence_pct",0)
            st.markdown(f"""
            <div class="result-card {sty['rc']}">
              <div class="rct">
                {sty['icon']} {r.get('location_name','')}
                <span style="float:right;font-size:0.68rem;color:#9aa0a6">{r.get('filename','')}</span>
              </div>
              <div style="font-size:0.72rem;color:#5f6368;margin-bottom:8px">
                📷 {r.get('camera_id','')} &nbsp;·&nbsp;
                📍 {r.get('latitude',0):.4f}, {r.get('longitude',0):.4f}
              </div>
              <div class="rcr">
                <div class="rcs">
                  <div class="rcn">{d:.0f}</div>
                  <div class="rcl">cm depth</div>
                </div>
                <div class="rcs">
                  <div class="rcn">{conf:.0f}%</div>
                  <div class="rcl">confidence</div>
                </div>
                <div class="rcs">
                  <div class="rcn" style="font-size:0.95rem;color:{sty['bg']}">{risk}</div>
                  <div class="rcl">risk level</div>
                </div>
              </div>
              <div class="rca">⚡ {r.get('recommended_action','Monitor situation')}</div>
            </div>""", unsafe_allow_html=True)

            if r.get("gemini_risk"):
                agreed = r.get("gemini_agreement")
                st.markdown(f"""
                <div class="gemini-strip">
                  <strong>🤖 Gemini</strong>: {r.get('gemini_risk','')} · {r.get('gemini_depth_cm',0)} cm
                  &nbsp; {'✅ Agreed' if agreed else '⚠️ Disagreed'}<br>
                  <span style="opacity:.8">{r.get('gemini_reasoning','')}</span>
                </div>""", unsafe_allow_html=True)

# ── Full results table ────────────────────────────────────────────────────────
if st.session_state.results:
    st.markdown('<div class="sec-lbl">📊 All Results</div>', unsafe_allow_html=True)
    df  = pd.DataFrame(st.session_state.results)
    cols = ["filename","camera_id","location_name","latitude","longitude",
            "water_depth_cm","risk_level","flood_detected","confidence_pct",
            "ensemble_method","timestamp"]
    df2 = df[[c for c in cols if c in df.columns]]

    def _colour_risk(val):
        c = {"NO FLOOD":"#e6f4ea","LOW RISK":"#e8f0fe","MODERATE":"#fef7e0",
             "HIGH RISK":"#fce8e6","CRITICAL":"#f5c6c6"}
        return f"background-color:{c.get(val,'')};font-weight:600"

    st.dataframe(
        df2.style.applymap(_colour_risk, subset=["risk_level"]),
        use_container_width=True,
        hide_index=True,
        height=200,
    )
    _, dl_col = st.columns([3,1])
    dl_col.download_button(
        "⬇️ Export CSV",
        df2.to_csv(index=False),
        f"floodwatch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        use_container_width=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 8px;color:#9aa0a6;font-size:0.73rem;
            border-top:1px solid #dadce0;margin-top:24px">
  FloodWatch AI v3 &nbsp;·&nbsp; SegFormer + YOLOv8 + Depth Anything V2 + Gemini Ensemble
  &nbsp;·&nbsp; BBMP Bengaluru
  &nbsp;·&nbsp; Map tiles © Google Maps
</div>
""", unsafe_allow_html=True)
