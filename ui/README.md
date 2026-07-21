# ui/ — Streamlit Web Apps

Two apps — one for uploading images to test, one for live monitoring.

| File | Purpose | Run command |
|------|---------|-------------|
| `upload_app.py` | Upload flood image → instant prediction + map pin | `streamlit run ui/upload_app.py` |
| `app.py` | Live dashboard — reads predictions from DB, shows map + charts | `streamlit run ui/app.py` |

## upload_app.py (Image Tester)
- Drag-and-drop any flood image
- Shows: flood detected, depth (cm), risk level, water coverage %
- Displays image with overlaid water mask
- Pins location on map if GPS metadata present

## app.py (Live Dashboard)
- Reads from PostgreSQL `predictions` table
- Auto-refreshes every 30 seconds
- Shows risk heatmap over Bengaluru
- Alerts panel for high/critical zones

## Requirements
```bash
pip install streamlit folium streamlit-folium pillow
```
