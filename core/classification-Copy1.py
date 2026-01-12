import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pygmt
import rioxarray
from scipy.ndimage import generic_filter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from .altimetry_extractors import (
    extract_altimetry_data,
    discover_altimetry_files
)

# ==========================================================
# RESOLUTION CHECK
# ==========================================================
def is_high_resolution(filename: str) -> bool:
    f = filename.upper()
    if "ENHANCED_MEASUREMENT" in f:
        return True
    if "SWOT" in f:
        return True
    if "S6" in f and "__HR_" in f:
        return True
    return False

# ==========================================================
# VECTOR LOADER
# ==========================================================
def load_vector(input_obj):
    if isinstance(input_obj, gpd.GeoDataFrame):
        gdf = input_obj.copy()
    elif isinstance(input_obj, str):
        gdf = gpd.read_file(input_obj)
    else:
        raise TypeError("Input must be filepath or GeoDataFrame")

    if gdf.crs is None:
        raise ValueError("Vector must have CRS")

    return gdf.to_crs("EPSG:4326")

# ==========================================================
# TERRAIN METRICS (CLASS 5)
# ==========================================================
def elevation_stats(arr):
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    return {
        "p95_p05": float(np.percentile(arr, 95) - np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95))
    }

def horn_slope_deg(dem_da):
    dx = 111320 * np.cos(np.deg2rad(float(dem_da.lat.mean())))
    dy = 111320

    z = np.where(np.isfinite(dem_da.values), dem_da.values, np.nan)
    zp = np.pad(z, 1, constant_values=np.nan)

    z1,z2,z3 = zp[:-2,:-2], zp[:-2,1:-1], zp[:-2,2:]
    z4,z6 = zp[1:-1,:-2], zp[1:-1,2:]
    z7,z8,z9 = zp[2:,:-2], zp[2:,1:-1], zp[2:,2:]

    dzdx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * dx)
    dzdy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8 * dy)

    return np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))

def riley_tri(arr):
    def tri(w):
        c = w[4]
        neigh = np.delete(w, 4)
        neigh = neigh[np.isfinite(neigh)]
        if not np.isfinite(c) or len(neigh) < 5:
            return np.nan
        return np.sqrt(np.sum((c - neigh)**2))
    return generic_filter(arr, tri, size=3, mode="constant", cval=np.nan)

def is_mountainous_reservoir(perm_water_input, buffer_km=5):
    perm = load_vector(perm_water_input).to_crs("EPSG:3857")
    perm["geometry"] = perm.geometry.buffer(buffer_km * 1000)
    buf = perm.to_crs("EPSG:4326")

    bounds = buf.total_bounds
    region = [bounds[0]-0.2, bounds[2]+0.2, bounds[1]-0.2, bounds[3]+0.2]

    dem = pygmt.datasets.load_earth_relief("03s", region=region).rio.write_crs("EPSG:4326")
    dem_clip = dem.rio.clip(buf.geometry, buf.crs)

    elev = dem_clip.values
    slope = horn_slope_deg(dem_clip)
    tri = riley_tri(elev)

    criteria = sum([
        elevation_stats(elev)["p95_p05"] >= 500,
        elevation_stats(slope)["p95"] >= 10,
        elevation_stats(tri)["p95"] >= 100
    ])

    return criteria >= 2

# ==========================================================
# ALTMMETRY CHECK (STRICT MAX → PERM)
# ==========================================================
def check_altimetry_tracks(base_folder, max_water_input, perm_water_input):
    max_water = load_vector(max_water_input)
    perm_water = load_vector(perm_water_input)

    records = []
    files = discover_altimetry_files(base_folder)

    for f in files:
        try:
            df = extract_altimetry_data(f)
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs="EPSG:4326"
            )

            # ---------- STRICT FILTER ----------
            max_hits = gpd.sjoin(gdf, max_water, predicate="intersects", how="inner")
            if max_hits.empty:
                continue  # 🔥 IGNORE FILE COMPLETELY

            perm_hits = gpd.sjoin(
                max_hits.drop(columns="index_right", errors="ignore"),
                perm_water,
                predicate="intersects",
                how="inner"
            )

            status = (
                "✅ Passes permanent water"
                if not perm_hits.empty
                else "❌ Seasonal only"
            )

            records.append({
                "filename": os.path.basename(f),
                "status": status
            })

        except Exception:
            continue

    return pd.DataFrame(records)

# ==========================================================
# POINT COLLECTION (MAX WATER ONLY)
# ==========================================================
def collect_points_over_max_water(base_folder, max_water_input):
    max_water = load_vector(max_water_input)
    files = discover_altimetry_files(base_folder)
    points = []

    for f in files:
        try:
            df = extract_altimetry_data(f)
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs="EPSG:4326"
            )

            hits = gpd.sjoin(gdf, max_water, predicate="intersects", how="inner")
            if not hits.empty:
                points.append(hits.drop(columns="index_right", errors="ignore"))

        except Exception:
            continue

    if not points:
        return None

    return gpd.GeoDataFrame(pd.concat(points, ignore_index=True), crs="EPSG:4326")

# ==========================================================
# PLOTTING
# ==========================================================
def plot_classification(max_water_input, perm_water_input, points, final_class):
    max_water = load_vector(max_water_input)
    perm_water = load_vector(perm_water_input)

    fig, ax = plt.subplots(figsize=(6, 6))

    max_water.plot(ax=ax, facecolor="lightblue", edgecolor="blue", alpha=0.4)
    perm_water.plot(ax=ax, facecolor="none", edgecolor="navy", linewidth=1)

    if points is not None and not points.empty:
        points.plot(ax=ax, color="gray", markersize=0.8, alpha=0.6)

    ax.set_title(f"Reservoir classification: Class {final_class}")
    ax.grid(alpha=0.3)
    plt.show()

# ==========================================================
# MASTER DRIVER
# ==========================================================
def run_full_classification(base_folder, max_water_input, perm_water_input):
    # ---------- CLASS 5 ----------
    if is_mountainous_reservoir(perm_water_input):
        pts = collect_points_over_max_water(base_folder, max_water_input)
        plot_classification(max_water_input, perm_water_input, pts, 5)
        return 5, None


    # ---------- CLASS 1–4 ----------
    results = check_altimetry_tracks(base_folder, max_water_input, perm_water_input)

    if results.empty:
        return None, results

    any_fail = results["status"].str.contains("❌").any()
    has_hr = results["filename"].apply(is_high_resolution).any()

    if not any_fail:
        final_class = 1 if has_hr else 2
    else:
        final_class = 3 if has_hr else 4

    pts = collect_points_over_max_water(base_folder, max_water_input)
    plot_classification(max_water_input, perm_water_input, pts, final_class)

    return final_class, results

