# ui/app.py
"""
FloodWatch AI — Streamlit Dashboard
=====================================
Completely decoupled from the pipeline.
Reads ONLY from PostgreSQL (db.postgres.PostgresWriter.latest_per_camera).
Auto-refreshes every 60 seconds to show new batch results.

Run: streamlit run ui/app.py
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import folium
import streamlit as st
from streamlit_folium import st_folium
from db.postgres import PostgresWriter, DB_URL
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FloodWatch AI — Bengaluru",
    page_icon="🌊",
    layout="wide",
)

RISK_STYLE = {
    "NO FLOOD":  {"bg": "#2ecc71", "icon": "✅", "fc": "green"},
    "LOW RISK":  {"bg": "#3498db", "icon": "🟦", "fc": "blue"},
    "MODERATE":  {"bg": "#f39c12", "icon": "🟠", "fc": "orange"},
    "HIGH RISK": {"bg": "#e74c3c", "icon": "🔴", "fc": "red"},
    "CRITICAL":  {"bg": "#7b241c", "icon": "🚨", "fc": "darkred"},
}

REFRESH_SECS = 60


# ── DB connection (cached per session) ───────────────────────────────────────
@st.cache_resource
def get_writer():
    return PostgresWriter(DB_URL)


def load_readings():
    return get_writer().latest_per_camera()


# ── Map builder ──────────────────────────────────────────────────────────────
def build_map(readings: list[dict]) -> folium.Map:
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=12,
                   tiles="CartoDB positron")
    for r in readings:
        risk = r.get("risk_level", "NO FLOOD")
        sty  = RISK_STYLE.get(risk, RISK_STYLE["NO FLOOD"])
        d    = r.get("water_depth_cm", 0)
        conf = r.get("confidence_pct", 0)

        popup = f"""
        <div style="font-family:Arial;min-width:250px;font-size:13px">
          <h4 style="margin:0 0 4px;color:{sty['bg']}">{sty['icon']} {risk}</h4>
          <hr style="margin:4px 0">
          <table style="width:100%">
            <tr><td>📷 Camera</td><td><b>{r.get('camera_id','')}</b></td></tr>
            <tr><td>📍 Location</td><td><b>{r.get('location_name','')}</b></td></tr>
            <tr style="background:#f8f9fa">
              <td><b>① Flood</b></td>
              <td><b>{"YES 🌊" if r.get('flood_detected') else "NO ✅"}</b></td></tr>
            <tr style="background:#f8f9fa">
              <td><b>② Depth</b></td><td><b>{d} cm</b></td></tr>
            <tr style="background:#f8f9fa">
              <td><b>③ Risk</b></td>
              <td><b style="color:{sty['bg']}">{risk}</b></td></tr>
            <tr style="background:#f8f9fa">
              <td><b>⑤ Confidence</b></td><td><b>{conf}%</b></td></tr>
          </table>
          <hr style="margin:4px 0">
          <div style="background:{sty['bg']};color:white;padding:6px;border-radius:4px;
                      font-size:12px">
            <b>④ Action:</b> {r.get('recommended_action','')}
          </div>
          <div style="color:#888;font-size:11px;margin-top:4px">
            🕐 {r.get('captured_at','')}<br>
            🧠 seg={r.get('seg_engine','')} yolo={r.get('yolo_engine','')}
               depth={r.get('depth_engine','')}
          </div>
        </div>"""

        folium.Marker(
            location=[r["latitude"], r["longitude"]],
            popup=folium.Popup(popup, max_width=300),
            tooltip=f"{sty['icon']} {r.get('location_name','')} — {d}cm",
            icon=folium.Icon(color=sty["fc"], icon="tint", prefix="fa"),
        ).add_to(m)

        if risk in ("CRITICAL", "HIGH RISK"):
            folium.CircleMarker(
                location=[r["latitude"], r["longitude"]],
                radius=28, color=sty["bg"], fill=False, weight=2, opacity=0.5,
            ).add_to(m)
    return m


# ─────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────
st.markdown("# 🌊 FloodWatch AI — Bengaluru Live Flood Map")
st.caption(f"Auto-refreshes every {REFRESH_SECS}s · Reads from PostgreSQL · "
           f"Last loaded: {datetime.now().strftime('%H:%M:%S')}")
st.divider()

readings = load_readings()

# KPI row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📷 Cameras",   len(readings))
c2.metric("⚠️ Floods",    sum(1 for r in readings if r.get("flood_detected")),   delta_color="inverse")
c3.metric("🚨 Critical",  sum(1 for r in readings if r.get("risk_level")=="CRITICAL"), delta_color="inverse")
c4.metric("💧 Max Depth", f"{max((r.get('water_depth_cm',0) for r in readings), default=0):.0f} cm")
c5.metric("🔄 Next Batch","~15 min")
st.divider()

# Map + table
map_col, tbl_col = st.columns([3, 2])

with map_col:
    st.subheader("🗺️ Bengaluru Live Map")
    if readings:
        st_folium(build_map(readings), width=700, height=500, returned_objects=[])
    else:
        st.info("No readings in database yet. Pipeline will populate after first batch.")

with tbl_col:
    st.subheader("📋 Latest per Camera")
    if readings:
        df = pd.DataFrame(readings)[[
            "camera_id","location_name","water_depth_cm","risk_level","confidence_pct","captured_at"
        ]]
        df.columns = ["Camera","Location","Depth(cm)","Risk","Conf%","Time"]

        def colour(val):
            c = {"NO FLOOD":"#d5f5e3","LOW RISK":"#d6eaf8",
                 "MODERATE":"#fdebd0","HIGH RISK":"#fadbd8","CRITICAL":"#f1948a"}
            return f"background-color:{c.get(val,'')}"

        st.dataframe(df.style.applymap(colour, subset=["Risk"]),
                     use_container_width=True, hide_index=True)

        csv = df.to_csv(index=False)
        st.download_button("⬇️ Export CSV", csv,
            f"flood_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
    else:
        st.caption("Waiting for first batch…")

# Explainability
st.divider()
st.subheader("🔍 Engine Breakdown (Explainability)")
if readings:
    ex_df = pd.DataFrame(readings)
    if "seg_engine" in ex_df.columns:
        for _, row in ex_df.iterrows():
            with st.expander(f"{row.get('location_name','')} — {row.get('risk_level','')}"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Seg Engine",   row.get("seg_engine",""))
                col2.metric("YOLO Engine",  row.get("yolo_engine",""))
                col3.metric("Depth Engine", row.get("depth_engine",""))
                col1.metric("Water Coverage", f"{row.get('water_coverage_pct',0):.1f}%")
                col2.metric("Mean Depth",     f"{row.get('mean_flood_depth_cm',0):.1f} cm")
                col3.metric("Calibration",    row.get("calibration_source",""))

# Auto-refresh
st.markdown(f"""
<script>
setTimeout(function(){{ window.location.reload(); }}, {REFRESH_SECS * 1000});
</script>
""", unsafe_allow_html=True)

st.divider()
st.caption("FloodWatch AI v3 | SegFormer + YOLOv8 + Depth Anything V2 | BBMP Bengaluru")
