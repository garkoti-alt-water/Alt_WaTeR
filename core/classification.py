import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt
import rioxarray

from shapely.geometry import Polygon
from scipy.ndimage import generic_filter

from .altimetry_extractors import (
    extract_altimetry_data,
    discover_altimetry_files
)

# ==========================================================
# GMT SAFETY CONFIG (CRITICAL)
# ==========================================================
pygmt.config(GMT_COMPATIBILITY="6")

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
# TERRAIN HELPERS
# ==========================================================
def horn_slope_deg(z, dx, dy):
    zp = np.pad(z, 1, constant_values=np.nan)

    z1, z2, z3 = zp[:-2, :-2], zp[:-2, 1:-1], zp[:-2, 2:]
    z4, z6 = zp[1:-1, :-2], zp[1:-1, 2:]
    z7, z8, z9 = zp[2:, :-2], zp[2:, 1:-1], zp[2:, 2:]

    dzdx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * dx)
    dzdy = ((z7 + 2*z8 + z9) - (z1 + 2*z2 + z3)) / (8 * dy)

    return np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))


def riley_tri(z):
    def tri_func(w):
        c = w[4]
        if not np.isfinite(c):
            return np.nan
        neigh = np.delete(w, 4)
        neigh = neigh[np.isfinite(neigh)]
        if neigh.size < 5:
            return np.nan
        return np.sqrt(np.sum((c - neigh)**2))

    return generic_filter(z, tri_func, size=3, mode="constant", cval=np.nan)


# ==========================================================
# TRACK-WISE BUFFER
# ==========================================================
def buffer_from_track_extent(points_inside_water, max_water_gdf, buffer_km):
    if points_inside_water.empty:
        raise RuntimeError("No points inside water")

    minx, miny, maxx, maxy = points_inside_water.total_bounds
    track_rect = Polygon([
        (minx, miny),
        (minx, maxy),
        (maxx, maxy),
        (maxx, miny)
    ])

    rect_gdf = gpd.GeoDataFrame(
        geometry=[track_rect],
        crs="EPSG:4326"
    )

    lon, lat = track_rect.centroid.xy
    utm_zone = int((lon[0] + 180) / 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"

    rect_utm = rect_gdf.to_crs(utm_crs)
    water_utm = max_water_gdf.to_crs(utm_crs)

    buffer_utm = rect_utm.buffer(buffer_km * 1000)
    buffer_land = buffer_utm.difference(water_utm.union_all())
    buffer_land = buffer_land.buffer(0)

    return gpd.GeoDataFrame(
        geometry=buffer_land,
        crs=utm_crs
    ).to_crs("EPSG:4326")


# ==========================================================
# TERRAIN + GEOID METRICS (FIXED)
# ==========================================================
def compute_terrain_indices_from_buffer(
    buffer_geom,
    dem_res="03s",
    geoid_res="01m"
):
    geom = buffer_geom.geometry if isinstance(buffer_geom, gpd.GeoDataFrame) else buffer_geom
    geom = geom.to_crs("EPSG:4326")

    bounds = geom.total_bounds
    region = [
        bounds[0] - 0.1,
        bounds[2] + 0.1,
        bounds[1] - 0.1,
        bounds[3] + 0.1,
    ]

    # ---------------- DEM (SAFE) ----------------
    dem = (
        pygmt.datasets.load_earth_relief(
            resolution=dem_res,
            region=region
        )
        .rio.write_crs("EPSG:4326")
        .load()
    )

    dem_clip = dem.rio.clip(geom, geom.crs)
    z = dem_clip.values.astype(float)

    zf = z[np.isfinite(z)]
    dz_p95_p05 = np.percentile(zf, 95) - np.percentile(zf, 5)

    lat_mean = float(dem_clip.lat.mean())
    dx = 111320 * np.cos(np.deg2rad(lat_mean))
    dy = 111320

    slope = horn_slope_deg(z, dx, dy)
    tri = riley_tri(z)

    slope_p95 = np.nanpercentile(slope, 95)
    tri_p95 = np.nanpercentile(tri, 95)

    # ---------------- GEOID (SAFE) ----------------
    geoid = (
        pygmt.datasets.load_earth_geoid(
            resolution=geoid_res,
            region=region
        )
        .rio.write_crs("EPSG:4326")
        .load()
    )

    geoid_clip = geoid.rio.clip(geom, geom.crs)
    N = geoid_clip.values.astype(float)
    Nf = N[np.isfinite(N)]
    geoid_p95_p05 = np.percentile(Nf, 95) - np.percentile(Nf, 5)

    return {
        "elev_p95_p05_m": dz_p95_p05,
        "slope_p95_deg": slope_p95,
        "tri_p95_m": tri_p95,
        "geoid_p95_p05_m": geoid_p95_p05,
    }


# ==========================================================
# ALTMMETRY CHECK
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

            max_hits = gpd.sjoin(gdf, max_water, predicate="intersects", how="inner")
            if max_hits.empty:
                continue

            perm_hits = gpd.sjoin(
                max_hits.drop(columns="index_right", errors="ignore"),
                perm_water,
                predicate="intersects",
                how="inner"
            )

            records.append({
                "filename": os.path.basename(f),
                "perm_hit": not perm_hits.empty
            })

        except Exception:
            continue

    return pd.DataFrame(records)


# ==========================================================
# TRACK-WISE TERRAIN COMPLEXITY
# ==========================================================
def is_complex_terrain(base_folder, max_water_input, buffer_km=5):
    max_water = load_vector(max_water_input)
    files = discover_altimetry_files(base_folder)

    for f in files:
        try:
            df = extract_altimetry_data(f)
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs="EPSG:4326"
            )

            hits = gpd.sjoin(
                gdf, max_water, predicate="intersects", how="inner"
            )
            if hits.empty:
                continue

            buffer_land = buffer_from_track_extent(
                hits, max_water, buffer_km
            )

            metrics = compute_terrain_indices_from_buffer(buffer_land)

            if (
                metrics["elev_p95_p05_m"] >= 500 or
                metrics["slope_p95_deg"] >= 10 or
                metrics["tri_p95_m"] >= 100 or
                metrics["geoid_p95_p05_m"] >= 1.5
            ):
                return True

        except Exception:
            continue

    return False


# ==========================================================
# MASTER DRIVER
# ==========================================================
def run_full_classification(base_folder, max_water_input, perm_water_input):

    results = check_altimetry_tracks(
        base_folder, max_water_input, perm_water_input
    )

    if results.empty:
        return None, results

    has_hr = results["filename"].apply(is_high_resolution).any()
    any_fail = (~results["perm_hit"]).any()

    if not any_fail:
        base_class = 1 if has_hr else 2
    else:
        base_class = 3 if has_hr else 4

    complex_terrain = is_complex_terrain(
        base_folder, max_water_input
    )

    suffix = "B" if complex_terrain else "A"
    final_class = f"{base_class}{suffix}"

    return final_class, results
