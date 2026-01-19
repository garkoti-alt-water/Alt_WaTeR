import os
import sys
from pathlib import Path

import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from shapely.geometry import box, shape

import folium
from streamlit_folium import st_folium
import ee

# -------------------------------------------------
# Ensure package imports work
# -------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from core.reservoir_pipeline import run_reservoir_analysis

# -------------------------------------------------
# CLASS COLORS (A/B aware)
# -------------------------------------------------
BASE_CLASS_COLORS = {
    1: "#a65628",
    2: "#245c52",
    3: "#CC79A7",
    4: "#c51b8a",
}

def get_class_color(reservoir_class):
    if isinstance(reservoir_class, str):
        base = int(reservoir_class[0])
        suffix = reservoir_class[1].upper()
    else:
        base = int(reservoir_class)
        suffix = "A"

    base_color = BASE_CLASS_COLORS.get(base, "#000000")

    if suffix == "A":
        return base_color
    else:
        import matplotlib.colors as mcolors
        rgb = mcolors.to_rgb(base_color)
        darker = tuple(max(0, c * 0.7) for c in rgb)
        return mcolors.to_hex(darker)

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(layout="centered")
st.title("Alt-WaTeR")
st.caption("Water level Time series generation using satellite Altimetry")
st.markdown("---")

# -------------------------------------------------
# Google Earth Engine
# -------------------------------------------------
st.subheader("Google Earth Engine")

gee_ready = False
try:
    ee.Initialize()
    gee_ready = True
    st.success("Google Earth Engine authenticated.")
except Exception:
    st.warning("Earth Engine not authenticated.")

    if st.button("Authenticate Google Earth Engine"):
        ee.Authenticate()
        ee.Initialize()
        st.success("Authentication successful. Please rerun.")
        st.stop()

st.markdown("---")

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# -------------------------------------------------
# AOI SELECTION
# -------------------------------------------------
st.subheader("Area of interest definition")

aoi_mode = st.radio(
    "Choose AOI definition method:",
    ["Bounding box", "Draw polygon"],
)

aoi_gdf = None

# -------------------------------------------------
# Bounding box
# -------------------------------------------------
if aoi_mode == "Bounding box":
    c1, c2 = st.columns(2)
    with c1:
        min_lon = st.text_input("Minimum longitude")
        min_lat = st.text_input("Minimum latitude")
    with c2:
        max_lon = st.text_input("Maximum longitude")
        max_lat = st.text_input("Maximum latitude")

    if min_lon and min_lat and max_lon and max_lat:
        bbox_geom = box(
            float(min_lon), float(min_lat),
            float(max_lon), float(max_lat),
        )
        aoi_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")

# -------------------------------------------------
# Polygon draw
# -------------------------------------------------
if aoi_mode == "Draw polygon":
    m = folium.Map(location=[20, 78], zoom_start=4)

    draw = folium.plugins.Draw(
        export=True,
        draw_options={
            "polygon": True,
            "rectangle": False,
            "circle": False,
            "polyline": False,
            "marker": False,
            "circlemarker": False,
        },
    )
    draw.add_to(m)

    map_data = st_folium(m, height=650, width=900)
    drawings = map_data.get("all_drawings", [])

    if drawings:
        geom = shape(drawings[0]["geometry"])
        aoi_gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")

# -------------------------------------------------
# Inputs
# -------------------------------------------------
st.markdown("---")
altimetry_folder = st.text_input("Altimetry data directory")
output_csv = st.text_input("Output CSV file")

run = st.button("Run analysis")

# -------------------------------------------------
# Run
# -------------------------------------------------
if run:
    with st.spinner("Processing altimetry data..."):
        result = run_reservoir_analysis(
            aoi_gdf=aoi_gdf,
            altimetry_folder=altimetry_folder,
            output_csv=output_csv,
        )

    st.session_state.result = result
    st.session_state.analysis_done = True

# -------------------------------------------------
# Results
# -------------------------------------------------
if st.session_state.analysis_done:

    result = st.session_state.result
    ts_df = result["timeseries"]
    perm_gdf = result["perm_gdf"]
    max_gdf = result["max_gdf"]
    reservoir_class = result["reservoir_class"]

    class_color = get_class_color(reservoir_class)

    st.success("Processing complete.")
    st.caption(f"Reservoir class: **{reservoir_class}**")

    # Time series
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(ts_df["date"], ts_df["elevation"], color=class_color, s=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Water level (m)")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # Map
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    perm_gdf.plot(ax=ax, facecolor="blue", alpha=0.9)
    max_gdf.plot(ax=ax, facecolor="lightblue", alpha=0.9)

    gpd.GeoDataFrame(
        ts_df,
        geometry=gpd.points_from_xy(ts_df.longitude, ts_df.latitude),
        crs="EPSG:4326",
    ).plot(ax=ax, color=class_color, markersize=4)

    ax.set_title(f"Reservoir class {reservoir_class}")
    ax.grid(alpha=0.3)
    st.pyplot(fig)
