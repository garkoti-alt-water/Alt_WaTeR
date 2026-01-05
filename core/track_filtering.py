import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Polygon

import ee
import geemap  

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .altimetry_extractors import extract_altimetry_data



def filter_tracks_by_boundary_and_max_extent(
    boundary,
    max_water_fc,
    base_folder,
    plot=True
):
    """
    Filter altimetry tracks such that:
      - Track intersects the given boundary (AOI)
      - ALL .nc files in the track pass through maximum water extent
        (after AOI filtering)
      - Uses full per-file extraction logic (not a representative file)

    Parameters
    ----------
    boundary : list or ee.Geometry.Polygon
        AOI polygon (lon/lat)
    max_water_fc : ee.FeatureCollection
        Maximum water extent polygons
    base_folder : str
        Root folder containing altimetry track folders
    plot : bool
        Plot accepted tracks on reservoir-style map

    Returns
    -------
    passing_tracks : list[str]
        Track folder paths that pass all checks
    passing_points_gdf : geopandas.GeoDataFrame
        All accepted altimetry points
    """

       # ----------------------------------------------------
    # 1. Boundary handling
    # ----------------------------------------------------
    if isinstance(boundary, list):
        boundary_geom = gpd.GeoSeries(
            [gpd.GeoSeries.from_xy(
                [c[0] for c in boundary],
                [c[1] for c in boundary]
            ).unary_union],
            crs="EPSG:4326"
        ).iloc[0]
    else:
        boundary_geom = geemap.ee_to_gdf(
            ee.FeatureCollection([ee.Feature(boundary)])
        ).geometry.iloc[0]

    # ----------------------------------------------------
    # 2. Maximum water extent geometry
    # ----------------------------------------------------
    max_gdf = geemap.ee_to_gdf(max_water_fc).to_crs(epsg=4326)

    if max_gdf.empty:
        raise ValueError("Maximum water extent FeatureCollection is empty.")

    try:
        max_union = max_gdf.geometry.union_all()
    except AttributeError:
        max_union = max_gdf.unary_union

    # ----------------------------------------------------
    # 3. Discover and group NetCDF files by track
    # ----------------------------------------------------
    nc_files = glob.glob(os.path.join(base_folder, "**/*.nc"), recursive=True)
    if not nc_files:
        raise RuntimeError(f"No NetCDF files found under {base_folder}")

    tracks = defaultdict(list)
    for f in nc_files:
        tracks[os.path.dirname(f)].append(f)

    passing_tracks = []
    passing_points = []

    # ----------------------------------------------------
    # 4. Track-level logic (FULL per-file processing)
    # ----------------------------------------------------
    for folder, files in tracks.items():

        folder_lower = folder.lower()

        # Sentinel-3: only enhanced_measurement.nc
        if "s3" in folder_lower or "sentinel-3" in folder_lower:
            files = [f for f in files if "enhanced_measurement.nc" in f.lower()]
            if not files:
                continue

        track_valid = True
        track_points = []

        for f in sorted(files):

            try:
                df = extract_altimetry_data(f)
            except Exception:
                track_valid = False
                break

            if "latitude" not in df or "longitude" not in df:
                track_valid = False
                break

            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs="EPSG:4326"
            )

            # --- AOI filtering ---
            gdf_aoi = gdf[gdf.geometry.within(boundary_geom)]
            if gdf_aoi.empty:
                continue  # file does not intersect AOI → ignore file

            # --- Max extent test ---
            if not gdf_aoi.geometry.within(max_union).all():
                track_valid = False
                break

            track_points.append(gdf_aoi)

        if track_valid and track_points:
            passing_tracks.append(folder)
            passing_points.extend(track_points)

    # ----------------------------------------------------
    # 5. Merge accepted points
    # ----------------------------------------------------
    if passing_points:
        passing_points_gdf = gpd.GeoDataFrame(
            pd.concat(passing_points, ignore_index=True),
            crs="EPSG:4326"
        )
    else:
        passing_points_gdf = gpd.GeoDataFrame(
            columns=["geometry"], crs="EPSG:4326"
        )

    # ----------------------------------------------------
    # 6. Plot accepted tracks on reservoir-style map
    # ----------------------------------------------------
    if plot and not passing_points_gdf.empty:
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        bounds = boundary_geom.bounds
        ax.set_extent(bounds, crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="none")
        ax.add_feature(cfeature.RIVERS, linewidth=0.5, alpha=0.5)

        # Maximum water extent (dark blue)
        max_gdf.plot(
            ax=ax,
            facecolor="#08519c",
            edgecolor="none",
            alpha=0.35,
            transform=ccrs.PlateCarree()
        )

        # Accepted tracks
        passing_points_gdf.plot(
            ax=ax,
            color="red",
            markersize=2,
            transform=ccrs.PlateCarree()
        )

        ax.set_title(
            "Altimetry tracks fully inside AOI and maximum water extent",
            fontsize=13,
            weight="bold"
        )

        plt.show()

    return passing_tracks, passing_points_gdf
