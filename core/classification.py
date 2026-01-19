import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pygmt
import rioxarray
import shapely

from scipy.ndimage import generic_filter
from .altimetry_extractors import extract_altimetry_data, discover_altimetry_files

pygmt.config(GMT_COMPATIBILITY="6")

# ==========================================================
# GEOMETRY SAFETY UTILITIES
# ==========================================================
def clean_geometries(gdf):
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notnull()]
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.is_valid & ~gdf.is_empty]
    return gdf


def safe_union(gdf):
    if gdf is None or gdf.empty:
        return None
    geoms = [
        g.buffer(0)
        for g in gdf.geometry
        if g and g.is_valid and not g.is_empty
    ]
    if not geoms:
        return None
    try:
        return shapely.union_all(geoms)
    except Exception:
        return None


# ==========================================================
# RESOLUTION CHECK
# ==========================================================
def is_high_resolution(filename: str) -> bool:
    f = filename.upper()
    return (
        "ENHANCED_MEASUREMENT" in f
        or "SWOT" in f
        or ("S6" in f and "__HR_" in f)
    )


# ==========================================================
# VECTOR LOADER
# ==========================================================
def load_vector(input_obj):
    gdf = (
        input_obj.copy()
        if isinstance(input_obj, gpd.GeoDataFrame)
        else gpd.read_file(input_obj)
    )
    return gdf.to_crs("EPSG:4326")


# ==========================================================
# INWARD BUFFER (−1 km)
# ==========================================================
def inward_buffer_km(gdf, buffer_km):
    gdf = clean_geometries(gdf)
    if gdf.empty:
        return gdf

    lon, _ = gdf.geometry.union_all().centroid.xy
    utm_zone = int((lon[0] + 180) / 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"

    gdf_utm = gdf.to_crs(utm_crs)
    buffered = gdf_utm.buffer(-buffer_km * 1000)
    buffered = buffered.buffer(0)
    buffered = buffered[~buffered.is_empty]

    return gpd.GeoDataFrame(
        geometry=buffered,
        crs=utm_crs
    ).to_crs("EPSG:4326")


# ==========================================================
# TERRAIN HELPERS
# ==========================================================
def horn_slope_deg(z, dx, dy):
    zp = np.pad(z, 1, constant_values=np.nan)
    dzdx = (
        (zp[:-2, 2:] + 2 * zp[1:-1, 2:] + zp[2:, 2:])
        - (zp[:-2, :-2] + 2 * zp[1:-1, :-2] + zp[2:, :-2])
    ) / (8 * dx)
    dzdy = (
        (zp[2:, :-2] + 2 * zp[2:, 1:-1] + zp[2:, 2:])
        - (zp[:-2, :-2] + 2 * zp[:-2, 1:-1] + zp[:-2, 2:])
    ) / (8 * dy)
    return np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))


def riley_tri(z):
    def tri(w):
        c = w[4]
        n = np.delete(w, 4)
        n = n[np.isfinite(n)]
        return np.sqrt(np.sum((c - n) ** 2)) if np.isfinite(c) and len(n) >= 5 else np.nan

    return generic_filter(z, tri, size=3, mode="constant", cval=np.nan)


# ==========================================================
# BUFFER FROM ALL TRACKS (ROBUST)
# ==========================================================
def buffer_from_all_tracks(points_inside_water, max_water_gdf, buffer_km):

    hull = points_inside_water.unary_union.convex_hull
    if hull.is_empty or not hull.is_valid:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    lon, _ = hull.centroid.xy
    utm_zone = int((lon[0] + 180) / 6) + 1
    utm_crs = f"EPSG:{32600 + utm_zone}"

    hull_utm = (
        gpd.GeoDataFrame(geometry=[hull], crs="EPSG:4326")
        .to_crs(utm_crs)
        .buffer(0)
    )

    water_utm = clean_geometries(max_water_gdf.to_crs(utm_crs))
    water_union = safe_union(water_utm)

    if water_union is None:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    try:
        buffer_land = (
            hull_utm.buffer(buffer_km * 1000)
            .difference(water_union)
            .buffer(0)
        )
    except Exception:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    if buffer_land.is_empty.all():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    return gpd.GeoDataFrame(
        geometry=buffer_land,
        crs=utm_crs
    ).to_crs("EPSG:4326")


# ==========================================================
# TERRAIN + GEOID METRICS (UPDATED: EXPLICIT FALLBACK)
# ==========================================================
def compute_terrain_indices_from_buffer(buffer_geom):
    geom = buffer_geom.to_crs("EPSG:4326")
    bounds = geom.total_bounds

    region = [
        bounds[0] - 0.1,
        bounds[2] + 0.1,
        bounds[1] - 0.1,
        bounds[3] + 0.1,
    ]

    dem = None
    dem_res = None

    for res in ["03s", "15s", "30s"]:
        try:
            dem = (
                pygmt.datasets.load_earth_relief(resolution=res, region=region)
                .rio.write_crs("EPSG:4326")
                .load()
            )
            dem_res = res
            break
        except Exception:
            continue

    if dem is None:
        raise RuntimeError("No earth_relief DEM available for this region")

    dem_clip = dem.rio.clip(geom.geometry, geom.crs)
    z = dem_clip.values.astype(float)

    lat_mean = float(dem_clip.lat.mean())
    dx = 111320 * np.cos(np.deg2rad(lat_mean))
    dy = 111320

    slope = horn_slope_deg(z, dx, dy)
    tri = riley_tri(z)

    geoid = (
        pygmt.datasets.load_earth_geoid("01m", region=region)
        .rio.write_crs("EPSG:4326")
        .load()
    )

    geoid_clip = geoid.rio.clip(geom.geometry, geom.crs)
    N = geoid_clip.values.astype(float)

    return {
        "dem_resolution": dem_res,
        "elev_p95_p05_m": float(np.nanpercentile(z, 95) - np.nanpercentile(z, 5)),
        "slope_p95_deg": float(np.nanpercentile(slope, 95)),
        "tri_p95_m": float(np.nanpercentile(tri, 95)),
        "geoid_p95_p05_m": float(np.nanpercentile(N, 95) - np.nanpercentile(N, 5)),
    }



# ==========================================================
# TERRAIN COMPLEXITY (NETCDF-SAFE)
# ==========================================================
def is_complex_terrain(base_folder, max_water_input):
    max_water = clean_geometries(load_vector(max_water_input))
    all_points = []

    for f in discover_altimetry_files(base_folder):
        try:
            df = extract_altimetry_data(f)
        except OSError:
            print(f"[WARN] Corrupted NetCDF skipped: {os.path.basename(f)}")
            try:
                os.remove(f)
            except Exception:
                pass
            continue
        except Exception:
            continue

        if df.empty:
            continue

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326",
        )
        all_points.append(gdf)

    if not all_points:
        return False, pd.DataFrame()

    pts_all = gpd.GeoDataFrame(
        pd.concat(all_points, ignore_index=True),
        crs="EPSG:4326",
    )

    inside = gpd.sjoin(
        pts_all, max_water, predicate="intersects", how="inner"
    )

    if inside.empty:
        return False, pd.DataFrame()

    buffer_land = buffer_from_all_tracks(inside, max_water, 5)
    if buffer_land.empty:
        return False, pd.DataFrame()

    metrics = compute_terrain_indices_from_buffer(buffer_land)

    failures = sum(
        [
            metrics["elev_p95_p05_m"] >= 500,
            metrics["slope_p95_deg"] >= 10,
            metrics["tri_p95_m"] >= 100,
            metrics["geoid_p95_p05_m"] >= 1.5,
        ]
    )

    return failures >= 2, pd.DataFrame([metrics])


# ==========================================================
# TRACK GEOMETRY CHECK (NETCDF-SAFE)
# ==========================================================
def check_altimetry_tracks(base_folder, max_water_input, perm_water_input):
    max_water = clean_geometries(load_vector(max_water_input))
    perm_water = clean_geometries(load_vector(perm_water_input))
    max_inner = inward_buffer_km(max_water, 1)

    records = []

    for f in discover_altimetry_files(base_folder):
        try:
            df = extract_altimetry_data(f)
        except OSError:
            print(f"[WARN] Corrupted NetCDF skipped: {os.path.basename(f)}")
            try:
                os.remove(f)
            except Exception:
                pass
            continue
        except Exception:
            continue

        if df.empty:
            continue

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326",
        )

        max_hits = gpd.sjoin(
            gdf, max_water, predicate="intersects", how="inner"
        )
        if max_hits.empty:
            continue

        perm_hits = gpd.sjoin(
            max_hits.drop(columns="index_right", errors="ignore"),
            perm_water,
            predicate="intersects",
            how="inner",
        )

        inner_hits = gpd.sjoin(
            perm_hits.drop(columns="index_right", errors="ignore"),
            max_inner,
            predicate="intersects",
            how="inner",
        )

        records.append(
            {
                "file": os.path.basename(f),
                "passes": (not perm_hits.empty) and (not inner_hits.empty),
            }
        )

    return pd.DataFrame(records)


# ==========================================================
# MASTER DRIVER
# ==========================================================
def run_full_classification(base_folder, max_water_input, perm_water_input):

    results = check_altimetry_tracks(
        base_folder, max_water_input, perm_water_input
    )

    if results.empty:
        print("⚠️  No altimetry track intersects maximum water extent — skipping")
        return None, None

    files = discover_altimetry_files(base_folder)
    has_hr = any(is_high_resolution(os.path.basename(f)) for f in files)

    any_fail = (~results["passes"]).any()

    if not any_fail:
        base_class = 1 if has_hr else 2
    else:
        base_class = 3 if has_hr else 4

    complex_terrain, terrain_metrics = is_complex_terrain(
        base_folder, max_water_input
    )

    suffix = "B" if complex_terrain else "A"
    final_class = f"{base_class}{suffix}"

    return final_class, results
