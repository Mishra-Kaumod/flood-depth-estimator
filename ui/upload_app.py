# ui/upload_app.py
"""
FloodWatch AI — Upload & Predict
Run: streamlit run ui/upload_app.py
"""

import os, sys, tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

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

# ── Global CSS — clean light theme ───────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #f7f8fa;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e8eaed;
}

/* ── Header strip ── */
.fw-header {
    background: linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%);
    color: #fff;
    padding: 18px 28px 14px;
    border-radius: 12px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.fw-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
.fw-header p  { margin: 2px 0 0; font-size: 0.85rem; opacity: .85; }

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 12px; margin-bottom: 20px; }
.kpi-card {
    flex: 1;
    background: #fff;
    border: 1px solid #e8eaed;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
}
.kpi-card .kpi-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1a73e8;
    line-height: 1;
}
.kpi-card .kpi-lbl {
    font-size: 0.72rem;
    color: #5f6368;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: .04em;
}
.kpi-card.warn .kpi-val  { color: #f29900; }
.kpi-card.crit .kpi-val  { color: #d93025; }
.kpi-card.good .kpi-val  { color: #1e8e3e; }

/* ── Risk badge ── */
.risk-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    color: #fff;
}

/* ── Result card ── */
.result-card {
    background: #fff;
    border: 1px solid #e8eaed;
    border-left: 4px solid #1a73e8;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.result-card.flood-yes { border-left-color: #d93025; }
.result-card.flood-low { border-left-color: #f29900; }
.result-card.flood-mod { border-left-color: #e37400; }
.result-card .rc-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #202124;
    margin-bottom: 8px;
}
.result-card .rc-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
}
.result-card .rc-stat { flex: 1; min-width: 80px; }
.result-card .rc-stat .rc-num {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1a73e8;
}
.result-card .rc-stat .rc-sub {
    font-size: 0.7rem;
    color: #5f6368;
    text-transform: uppercase;
}
.result-card .rc-action {
    margin-top: 8px;
    font-size: 0.8rem;
    color: #3c4043;
    background: #f1f3f4;
    border-radius: 6px;
    padding: 6px 10px;
}

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label {
    font-size: 0.8rem !important;
    color: #3c4043 !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stButton > button {
    border-radius: 8px;
    font-weight: 600;
}

/* ── Section title ── */
.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #5f6368;
    margin: 16px 0 8px;
}

/* ── Table ── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
RISK_STYLE = {
    "NO FLOOD":  {"bg": "#1e8e3e", "light": "#e6f4ea", "icon": "✅", "fc": "green",   "cls": "good"},
    "LOW RISK":  {"bg": "#1a73e8", "light": "#e8f0fe", "icon": "💧", "fc": "blue",    "cls": ""},
    "MODERATE":  {"bg": "#e37400", "light": "#fef7e0", "icon": "🟠", "fc": "orange",  "cls": "warn"},
    "HIGH RISK": {"bg": "#d93025", "light": "#fce8e6", "icon": "🔴", "fc": "red",     "cls": "crit"},
    "CRITICAL":  {"bg": "#6b1a1a", "light": "#fce8e6", "icon": "🚨", "fc": "darkred", "cls": "crit"},
}
BORDER_CLS = {"NO FLOOD":"","LOW RISK":"","MODERATE":"flood-mod","HIGH RISK":"flood-yes","CRITICAL":"flood-yes"}

CAMERAS = {
    "Silk Board Junction":     {"lat": 12.9172, "lon": 77.6228},
    "Bellandur Lake Road":     {"lat": 12.9254, "lon": 77.6784},
    "Marathahalli Bridge":     {"lat": 12.9563, "lon": 77.7010},
    "Hebbal Flyover":          {"lat": 13.0351, "lon": 77.5975},
    "Ulsoor Road":             {"lat": 12.9784, "lon": 77.6206},
    "Whitefield Main Road":    {"lat": 12.9698, "lon": 77.7500},
    "Koramangala 5th Block":   {"lat": 12.9352, "lon": 77.6245},
    "Custom ▸ enter manually": {"lat": 12.9716, "lon": 77.5946},
}

if "results" not in st.session_state:
    st.session_state.results = []

# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_pipeline(gemini_key: str = ""):
    cfg = {"pipeline": {"device": "cpu",
                        "gemini_api_key": gemini_key or os.environ.get("GEMINI_API_KEY", "")}}
    return PipelineRunner(cfg) if _HAS_PIPELINE else None


def run_on_image(img_bgr, camera_id, location_id, lat, lon, loc_name, gemini_key):
    pipeline = get_pipeline(gemini_key)
    if pipeline is None:
        return _heuristic_fallback(img_bgr, camera_id, location_id, lat, lon, loc_name)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        cv2.imwrite(f.name, img_bgr)
        tmp_path = Path(f.name)
    cam_img = CameraImage(image_path=tmp_path, camera_id=camera_id,
                          location_id=location_id, latitude=lat, longitude=lon,
                          location_name=loc_name, captured_at=datetime.now().isoformat())
    pred = pipeline.run_image(cam_img, batch_id="upload")
    tmp_path.unlink(missing_ok=True)
    return pred.__dict__


def _heuristic_fallback(img_bgr, camera_id, location_id, lat, lon, loc_name):
    lower = img_bgr[int(img_bgr.shape[0] * 0.5):, :]
    hsv_l = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
    blue  = cv2.inRange(hsv_l, np.array([85,20,30]),  np.array([135,255,255]))
    dark  = cv2.inRange(hsv_l, np.array([0,0,20]),    np.array([180,60,150]))
    pct   = np.count_nonzero(cv2.bitwise_or(blue, dark)) / (lower.shape[0] * lower.shape[1]) * 100
    depth = min(round(pct * 1.8, 1), 120.0)
    risk  = ["NO FLOOD","LOW RISK","MODERATE","HIGH RISK","CRITICAL"][min(4, int(depth // 15))]
    return {"camera_id": camera_id, "location_id": location_id, "latitude": lat,
            "longitude": lon, "location_name": loc_name, "water_depth_cm": depth,
            "risk_level": risk, "flood_detected": depth > 0, "confidence_pct": 55.0,
            "recommended_action": "Monitor situation", "ensemble_method": "heuristic",
            "gemini_risk": None, "gemini_depth_cm": None, "gemini_reasoning": None,
            "gemini_agreement": None, "timestamp": datetime.now().isoformat()}


def build_map(results):
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12, tiles="CartoDB positron")
    for r in results:
        risk = r.get("risk_level", "NO FLOOD")
        sty  = RISK_STYLE.get(risk, RISK_STYLE["NO FLOOD"])
        d    = r.get("water_depth_cm", 0)
        popup_html = f"""
        <div style="font-family:'Inter','Segoe UI',sans-serif;min-width:220px;padding:4px">
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
            <span style="font-size:1.1rem">{sty['icon']}</span>
            <strong style="color:{sty['bg']};font-size:0.95rem">{risk}</strong>
          </div>
          <table style="width:100%;border-collapse:collapse;font-size:0.82rem">
            <tr><td style="color:#5f6368;padding:2px 0">Flood detected</td>
                <td style="font-weight:600">{"Yes 🌊" if r.get("flood_detected") else "No ✅"}</td></tr>
            <tr><td style="color:#5f6368;padding:2px 0">Water depth</td>
                <td style="font-weight:600">{d} cm</td></tr>
            <tr><td style="color:#5f6368;padding:2px 0">Confidence</td>
                <td style="font-weight:600">{r.get("confidence_pct",0):.0f}%</td></tr>
            <tr><td style="color:#5f6368;padding:2px 0">Action</td>
                <td style="font-weight:600;color:{sty['bg']}">{r.get("recommended_action","")}</td></tr>
          </table>
          <div style="margin-top:8px;font-size:0.72rem;color:#80868b">
            {r.get("ensemble_method","model")} · {r.get("location_name","")}
          </div>
        </div>"""
        folium.Marker(
            location=[r["latitude"], r["longitude"]],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{sty['icon']} {r.get('location_name','')} — {d} cm",
            icon=folium.Icon(color=sty["fc"], icon="tint", prefix="fa"),
        ).add_to(m)
        if risk in ("CRITICAL", "HIGH RISK"):
            folium.CircleMarker([r["latitude"], r["longitude"]],
                                radius=30, color=sty["bg"], fill=True,
                                fill_color=sty["bg"], fill_opacity=0.07,
                                weight=2, opacity=0.4).add_to(m)
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 4px;display:flex;align-items:center;gap:10px">
      <span style="font-size:1.6rem">🌊</span>
      <div>
        <div style="font-weight:700;font-size:1rem;color:#202124">FloodWatch AI</div>
        <div style="font-size:0.72rem;color:#5f6368">BBMP Bengaluru · v2.1</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-title">📷 Upload Images</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-title">📍 Camera Location</div>', unsafe_allow_html=True)
    cam_sel = st.selectbox("Location", list(CAMERAS.keys()), label_visibility="collapsed")
    if cam_sel == "Custom ▸ enter manually":
        c1, c2 = st.columns(2)
        lat      = c1.number_input("Lat",  value=12.9716, format="%.4f")
        lon      = c2.number_input("Lon",  value=77.5946, format="%.4f")
        loc_name = st.text_input("Name", value="Custom Location")
    else:
        lat, lon = CAMERAS[cam_sel]["lat"], CAMERAS[cam_sel]["lon"]
        loc_name = cam_sel
        st.caption(f"📌 {lat:.4f}, {lon:.4f}")

    with st.expander("⚙️ Advanced"):
        location_id = st.text_input("Location ID", value="LOC_001")
        camera_id   = st.text_input("Camera ID",   value="CAM_001")
        gemini_key  = st.text_input("Gemini API Key", type="password",
                                    value=os.environ.get("GEMINI_API_KEY", ""),
                                    help="Optional — enables ensemble validation")
        if gemini_key:
            st.success("Gemini ensemble ON ✅", icon="🤖")

    st.markdown("")
    run_btn = st.button("▶  Run Prediction", type="primary", use_container_width=True)
    if st.button("🗑️  Clear results", use_container_width=True):
        st.session_state.results = []
        st.rerun()

    st.divider()
    st.markdown('<div class="section-title">Risk Legend</div>', unsafe_allow_html=True)
    for rk, sty in RISK_STYLE.items():
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
            f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
            f'background:{sty["bg"]}"></span>'
            f'<span style="font-size:0.8rem;color:#3c4043">{rk}</span></div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
gemini_key = gemini_key if "gemini_key" in dir() else ""

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="fw-header">
  <span style="font-size:2rem">🌊</span>
  <div>
    <h1>FloodWatch AI</h1>
    <p>Upload camera images · 5-stage AI pipeline · Real-time Bengaluru flood map</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────────
rds = st.session_state.results
n_floods   = sum(1 for r in rds if r.get("flood_detected"))
n_critical = sum(1 for r in rds if r.get("risk_level") == "CRITICAL")
max_depth  = max((r.get("water_depth_cm", 0) for r in rds), default=0)
avg_conf   = (sum(r.get("confidence_pct", 0) for r in rds) / len(rds)) if rds else 0

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-val">{len(rds)}</div>
    <div class="kpi-lbl">Images processed</div>
  </div>
  <div class="kpi-card {'warn' if n_floods else 'good'}">
    <div class="kpi-val">{n_floods}</div>
    <div class="kpi-lbl">Floods detected</div>
  </div>
  <div class="kpi-card {'crit' if n_critical else ''}">
    <div class="kpi-val">{n_critical}</div>
    <div class="kpi-lbl">Critical zones</div>
  </div>
  <div class="kpi-card {'warn' if max_depth > 30 else ''}">
    <div class="kpi-val">{max_depth:.0f} cm</div>
    <div class="kpi-lbl">Max water depth</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">{'🤖' if gemini_key else '—'}</div>
    <div class="kpi-lbl">Gemini ensemble</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Run predictions ───────────────────────────────────────────────────────────
if uploaded_files and run_btn:
    prog = st.progress(0, text="Initialising pipeline…")
    for i, uf in enumerate(uploaded_files):
        img_bytes = np.frombuffer(uf.read(), np.uint8)
        img_bgr   = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        cam_id    = f"{camera_id}_{i+1:02d}" if len(uploaded_files) > 1 else camera_id
        with st.spinner(f"Analysing {uf.name}…"):
            result = run_on_image(img_bgr, cam_id, location_id, lat, lon, loc_name, gemini_key)
            result["filename"] = uf.name
        st.session_state.results = [r for r in st.session_state.results
                                     if r.get("camera_id") != cam_id]
        st.session_state.results.append(result)
        prog.progress((i + 1) / len(uploaded_files), f"Done {i+1} / {len(uploaded_files)}")
    st.rerun()

elif run_btn and not uploaded_files:
    st.warning("Please upload at least one image first.", icon="⚠️")

# ── Map + Results two-column layout ──────────────────────────────────────────
map_col, res_col = st.columns([3, 2], gap="medium")

with map_col:
    st.markdown('<div class="section-title">🗺️ Bengaluru Flood Map</div>', unsafe_allow_html=True)
    st_folium(
        build_map(st.session_state.results),
        use_container_width=True,
        height=460,
        returned_objects=[],
    )

with res_col:
    st.markdown('<div class="section-title">📋 Prediction Results</div>', unsafe_allow_html=True)

    if not st.session_state.results:
        st.markdown("""
        <div style="text-align:center;padding:48px 16px;color:#9aa0a6">
          <div style="font-size:2.5rem">📷</div>
          <div style="margin-top:8px;font-size:0.85rem">
            Upload images and click <strong>Run Prediction</strong>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        for r in reversed(st.session_state.results):
            risk = r.get("risk_level", "NO FLOOD")
            sty  = RISK_STYLE.get(risk, RISK_STYLE["NO FLOOD"])
            cls  = BORDER_CLS.get(risk, "")
            d    = r.get("water_depth_cm", 0)
            conf = r.get("confidence_pct", 0)
            st.markdown(f"""
            <div class="result-card {cls}">
              <div class="rc-title">
                {sty['icon']} {r.get('location_name','')}
                <span style="float:right;font-size:0.72rem;color:#9aa0a6">{r.get('filename','')}</span>
              </div>
              <div class="rc-row">
                <div class="rc-stat">
                  <div class="rc-num">{d:.0f}</div>
                  <div class="rc-sub">cm depth</div>
                </div>
                <div class="rc-stat">
                  <div class="rc-num">{conf:.0f}%</div>
                  <div class="rc-sub">confidence</div>
                </div>
                <div class="rc-stat">
                  <div class="rc-num" style="font-size:1rem;color:{sty['bg']}">{risk}</div>
                  <div class="rc-sub">risk level</div>
                </div>
              </div>
              <div class="rc-action">⚡ {r.get('recommended_action','Monitor situation')}</div>
            </div>""", unsafe_allow_html=True)

            if r.get("gemini_risk"):
                agree = r.get("gemini_agreement")
                st.markdown(f"""
                <div style="background:#f8f9fa;border:1px solid #e8eaed;border-radius:6px;
                            padding:8px 12px;margin-top:-6px;margin-bottom:10px;font-size:0.8rem">
                  <strong>🤖 Gemini</strong>: {r.get('gemini_risk','')} · {r.get('gemini_depth_cm',0)} cm
                  {'  ✅ Agreed' if agree else '  ⚠️ Disagreed'}<br>
                  <span style="color:#5f6368">{r.get('gemini_reasoning','')}</span>
                </div>""", unsafe_allow_html=True)

# ── Full results table ────────────────────────────────────────────────────────
if st.session_state.results:
    st.markdown('<div class="section-title">📊 All Results</div>', unsafe_allow_html=True)
    import pandas as pd
    df  = pd.DataFrame(st.session_state.results)
    cols = ["filename","location_name","water_depth_cm","risk_level",
            "flood_detected","confidence_pct","ensemble_method","timestamp"]
    df2 = df[[c for c in cols if c in df.columns]]
    st.dataframe(df2, use_container_width=True, hide_index=True, height=200)
    c1, c2 = st.columns([3, 1])
    c2.download_button(
        "⬇️ Export CSV",
        df2.to_csv(index=False),
        f"floodwatch_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv",
        use_container_width=True,
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:24px 0 8px;color:#9aa0a6;font-size:0.75rem;border-top:1px solid #e8eaed;margin-top:24px">
  FloodWatch AI &nbsp;·&nbsp; SegFormer + YOLOv8 + Depth Anything V2 + Gemini Ensemble
  &nbsp;·&nbsp; BBMP Bengaluru
</div>
""", unsafe_allow_html=True)
