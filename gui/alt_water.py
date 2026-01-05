import os
import sys
from pathlib import Path
from io import BytesIO

import streamlit as st
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import ee

# -------------------------------------------------
# Ensure package imports work in Streamlit
# -------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from core.reservoir_pipeline import run_reservoir_analysis

# -------------------------------------------------
# Class colors (contrast with water masks, color-blind safe)
# -------------------------------------------------
CLASS_COLORS = {
    1: "#E69F00",  # orange
    2: "#D55E00",  # vermillion
    3: "#CC79A7",  # purple
    4: "#F0E442",  # yellow
    5: "#7F3C8D",  # dark violet
}

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(layout="centered")
st.title("Alt-WaTER")
st.caption(
    "Water level time series estimation for reservoirs using satellite altimetry"
)
st.markdown("---")

# -------------------------------------------------
# Google Earth Engine authentication
# -------------------------------------------------
st.subheader("Google Earth Engine")

gee_ready = False
try:
    ee.Initialize()
    gee_ready = True
    st.success("Google Earth Engine is authenticated.")
except Exception:
    st.warning(
        "Google Earth Engine is not authenticated.\n\n"
        "Authentication is required once per machine."
    )

    if st.button("Authenticate Google Earth Engine"):
        try:
            ee.Authenticate()
            ee.Initialize()
            st.success("Authentication successful. Please rerun the analysis.")
            st.stop()
        except Exception as e:
            st.error("Earth Engine authentication failed.")
            st.exception(e)
            st.stop()

st.markdown("---")

# -------------------------------------------------
# Session state initialization
# -------------------------------------------------
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# -------------------------------------------------
# Input section
# -------------------------------------------------
with st.form("inputs"):

    st.subheader("Reservoir extent")

    c1, c2 = st.columns(2)
    with c1:
        min_lon_str = st.text_input("Minimum longitude")
        min_lat_str = st.text_input("Minimum latitude")
    with c2:
        max_lon_str = st.text_input("Maximum longitude")
        max_lat_str = st.text_input("Maximum latitude")

    st.subheader("Data locations")
    altimetry_folder = st.text_input("Altimetry data directory")
    output_csv = st.text_input("Output CSV file")

    run = st.form_submit_button("Run analysis")

# -------------------------------------------------
# Run pipeline (ONLY when button is pressed)
# -------------------------------------------------
if run:

    if not gee_ready:
        st.error("Google Earth Engine is not authenticated.")
        st.stop()

    try:
        # ---- Parse inputs
        min_lon = float(min_lon_str)
        min_lat = float(min_lat_str)
        max_lon = float(max_lon_str)
        max_lat = float(max_lat_str)

        if min_lon >= max_lon or min_lat >= max_lat:
            st.error("Invalid bounding box definition.")
            st.stop()

        altimetry_folder = os.path.abspath(
            os.path.expanduser(altimetry_folder.strip().strip('"').strip("'"))
        )
        output_csv = os.path.abspath(
            os.path.expanduser(output_csv.strip().strip('"').strip("'"))
        )

        if not os.path.isdir(altimetry_folder):
            st.error("Altimetry directory does not exist.")
            st.stop()

        out_dir = os.path.dirname(output_csv)
        if out_dir and not os.path.isdir(out_dir):
            st.error("Output directory does not exist.")
            st.stop()

        # ---- Run analysis
        with st.spinner("Processing altimetry data..."):
            result = run_reservoir_analysis(
                min_lon=min_lon,
                min_lat=min_lat,
                max_lon=max_lon,
                max_lat=max_lat,
                altimetry_folder=altimetry_folder,
                output_csv=output_csv,
            )

        st.session_state.result = result
        st.session_state.analysis_done = True

    except Exception as e:
        st.error("Processing failed.")
        st.exception(e)

# -------------------------------------------------
# Display results (persistent across reruns)
# -------------------------------------------------
if st.session_state.analysis_done:

    result = st.session_state.result

    st.success("Processing complete.")
    st.caption(f"CSV written to: {result['output_csv']}")

    ts_df = result["timeseries"]
    perm_gdf = result["perm_gdf"]
    max_gdf = result["max_gdf"]
    reservoir_class = result["reservoir_class"]

    class_color = CLASS_COLORS.get(reservoir_class, "#000000")

# -------------------------------------------------
# Time series (PLOTTED IN GUI WITH CLASS COLOR)
# -------------------------------------------------
    st.markdown("---")
    st.subheader("Water level time series")

    fig_ts, ax = plt.subplots(figsize=(10, 4))

    ax.scatter(
        ts_df["date"],
        ts_df["elevation"],
        marker="o",
        s=12,
        color=class_color,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Water level (m)")
    ax.grid(alpha=0.3)

    st.pyplot(fig_ts)

    buf_ts = BytesIO()
    fig_ts.savefig(buf_ts, dpi=300, bbox_inches="tight")
    buf_ts.seek(0)

    st.download_button(
        "Download timeseries",
        data=buf_ts,
        file_name="altwater_timeseries.png",
        mime="image/png",
    )


    # -------------------------------------------------
    # Map
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("Reservoir map")

    fig_map, ax = plt.subplots(figsize=(6.5, 6.5))

    perm_gdf.plot(
        ax=ax,
        facecolor="lightgreen",
        edgecolor="green",
        alpha=0.6,
    )

    max_gdf.plot(
        ax=ax,
        facecolor="lightblue",
        edgecolor="blue",
        alpha=0.4,
    )

    gpd.GeoDataFrame(
        ts_df,
        geometry=gpd.points_from_xy(ts_df.longitude, ts_df.latitude),
        crs="EPSG:4326",
    ).plot(
        ax=ax,
        color=class_color,
        markersize=6,
    )

    legend_elements = [
        Patch(facecolor="lightgreen", edgecolor="green", label="Permanent water"),
        Patch(facecolor="lightblue", edgecolor="blue", label="Maximum water extent"),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor=class_color,
            markersize=6,
            label=f"Altimetry points (Class {reservoir_class})",
        ),
    ]

    ax.legend(handles=legend_elements, fontsize=8, loc="best")
    ax.set_title(f"Reservoir class {reservoir_class}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.3)

    st.pyplot(fig_map)

    buf_map = BytesIO()
    fig_map.savefig(buf_map, dpi=300, bbox_inches="tight")
    buf_map.seek(0)

    st.download_button(
        "Download map",
        data=buf_map,
        file_name=f"altwater_map_class_{reservoir_class}.png",
        mime="image/png",
    )
