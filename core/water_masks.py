import ee
import geemap
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.patches import Patch


# --------------------------------------------------
# Helper: merge polygons and remove holes
# --------------------------------------------------
def merge_and_fill(gdf):
    """
    Merge all geometries into a single polygon and remove holes.
    """
    if gdf.empty:
        return gdf

    geom = gdf.geometry.unary_union
    geom = geom.buffer(0)  # fix topology

    if isinstance(geom, Polygon):
        geom = Polygon(geom.exterior)
    elif isinstance(geom, MultiPolygon):
        geom = MultiPolygon([Polygon(p.exterior) for p in geom.geoms])

    return gpd.GeoDataFrame(geometry=[geom], crs=gdf.crs)

import ee
import geemap

# ----------------------------------------------------
# STEP 1: Authenticate Google Earth Engine
# ----------------------------------------------------
# This will open a browser window (or show a link in Colab)
# where the user must:
#   1. Log in with their Google account
#   2. Grant Earth Engine permissions
#   3. Copy the authorization code
#   4. Paste it back into the prompt
#
# NOTE:
# - This step is REQUIRED only the first time
# - After successful authentication, credentials
#   are saved locally and reused automatically
#
ee.Authenticate()

# ----------------------------------------------------
# STEP 2: Initialize Earth Engine
# ----------------------------------------------------
# Initializes the Earth Engine API using the authenticated
# user account.
#
# No project ID is required for most use cases.
# If the user has a registered GEE Cloud Project,
# it can be added as:
# ee.Initialize(project='your-project-id')
#
ee.Initialize()



# --------------------------------------------------
# Main function
# --------------------------------------------------
def water_masks(
    min_lon,
    min_lat,
    max_lon,
    max_lat,
    occurrence_threshold=90,
    scale=30,
    plot=True,
):
    """
    Identify a reservoir using HydroLAKES and extract
    permanent water and maximum water extent masks
    from JRC Global Surface Water (GSW), using a bounding box.

    Parameters
    ----------
    min_lon, min_lat, max_lon, max_lat : float
        Bounding box coordinates
    occurrence_threshold : int
        Permanent water occurrence threshold (%)
    scale : int
        Pixel resolution in meters
    plot : bool
        Plot results

    Returns
    -------
    perm_gdf : geopandas.GeoDataFrame
        Single merged permanent water polygon (no holes)
    max_gdf : geopandas.GeoDataFrame
        Single merged maximum water extent polygon (no holes)
    """

    # --------------------------------------------------
    # 1. AOI geometry (bounding box)
    # --------------------------------------------------
    aoi = ee.Geometry.Polygon([[
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]])

    # --------------------------------------------------
    # 2. Load HydroLAKES
    # --------------------------------------------------
    hydrolakes = ee.FeatureCollection(
        "projects/sat-io/open-datasets/HydroLAKES"
    )

    lakes = hydrolakes.filterBounds(aoi)

    if lakes.size().getInfo() == 0:
        raise ValueError("No HydroLAKES reservoir found inside bounding box.")

    # Select largest reservoir intersecting AOI
    reservoir = ee.Feature(
        lakes.sort("Lake_area", False).first()
    )

    geom = reservoir.geometry()

    # --------------------------------------------------
    # 3. Global Surface Water masks (CRITICAL: selfMask)
    # --------------------------------------------------
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")

    perm_mask = (
        gsw.select("occurrence")
        .gte(occurrence_threshold)
        .selfMask()
        .clip(geom)
    )

    max_mask = (
        gsw.select("max_extent")
        .eq(1)
        .selfMask()
        .clip(geom)
    )

    # --------------------------------------------------
    # 4. Raster → Vector
    # --------------------------------------------------
    perm_fc = perm_mask.reduceToVectors(
        geometry=geom,
        scale=scale,
        geometryType="polygon",
        eightConnected=True,
        maxPixels=1e13,
    )

    max_fc = max_mask.reduceToVectors(
        geometry=geom,
        scale=scale,
        geometryType="polygon",
        eightConnected=True,
        maxPixels=1e13,
    )

    # --------------------------------------------------
    # 5. Convert to GeoDataFrame
    # --------------------------------------------------
    perm_gdf = geemap.ee_to_gdf(perm_fc).set_crs("EPSG:4326")
    max_gdf = geemap.ee_to_gdf(max_fc).set_crs("EPSG:4326")

    # --------------------------------------------------
    # 6. Merge polygons & remove holes
    # --------------------------------------------------
    perm_gdf = merge_and_fill(perm_gdf)
    max_gdf = merge_and_fill(max_gdf)

    # --------------------------------------------------
    # 7. Plot
    # --------------------------------------------------
    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))

        max_gdf.plot(ax=ax, color="lightblue", edgecolor="blue", alpha=0.4)
        perm_gdf.plot(ax=ax,facecolor="none",edgecolor="navy", linewidth=1)

        legend_elements = [
            Patch(facecolor="lightblue", edgecolor="black", label="Maximum water extent"),
            Patch(facecolor="blue", edgecolor="black", label="Permanent water"),
        ]

        ax.legend(handles=legend_elements, loc="best", frameon=True,fontsize=9)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("GSW reservoir water masks")

        plt.show()

    return perm_gdf, max_gdf
