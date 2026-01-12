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


# --------------------------------------------------
# MAIN FUNCTION (UPDATED)
# --------------------------------------------------
def water_masks(
    aoi_gdf: gpd.GeoDataFrame,
    occurrence_threshold: int = 90,
    scale: int = 30,
    plot: bool = True,
):
    """
    Extract permanent water and maximum water extent masks
    from JRC Global Surface Water (GSW), using a polygon AOI.

    Parameters
    ----------
    aoi_gdf : geopandas.GeoDataFrame
        Area of interest (Polygon or MultiPolygon, EPSG:4326)
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
    # Earth Engine init (safe)
    # --------------------------------------------------
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

    # --------------------------------------------------
    # AOI validation
    # --------------------------------------------------
    if aoi_gdf.empty:
        raise ValueError("AOI GeoDataFrame is empty")

    if aoi_gdf.crs is None:
        raise ValueError("AOI GeoDataFrame has no CRS")

    if aoi_gdf.crs.to_string() != "EPSG:4326":
        aoi_gdf = aoi_gdf.to_crs("EPSG:4326")

    # --------------------------------------------------
    # Convert AOI to Earth Engine geometry
    # --------------------------------------------------
    aoi_ee = geemap.geopandas_to_ee(aoi_gdf)

    # --------------------------------------------------
    # Global Surface Water (GSW)
    # --------------------------------------------------
    gsw = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")

    perm_mask = (
        gsw.select("occurrence")
        .gte(occurrence_threshold)
        .selfMask()
        .clip(aoi_ee)
    )

    max_mask = (
        gsw.select("max_extent")
        .eq(1)
        .selfMask()
        .clip(aoi_ee)
    )

    # --------------------------------------------------
    # Raster → Vector
    # --------------------------------------------------
    perm_fc = perm_mask.reduceToVectors(
        geometry=aoi_ee,
        scale=scale,
        geometryType="polygon",
        eightConnected=True,
        maxPixels=1e13,
    )

    max_fc = max_mask.reduceToVectors(
        geometry=aoi_ee,
        scale=scale,
        geometryType="polygon",
        eightConnected=True,
        maxPixels=1e13,
    )

    # --------------------------------------------------
    # Convert to GeoDataFrame
    # --------------------------------------------------
    perm_gdf = geemap.ee_to_gdf(perm_fc)
    max_gdf  = geemap.ee_to_gdf(max_fc)

    if perm_gdf.empty or max_gdf.empty:
        raise RuntimeError("GSW returned empty water masks")

    perm_gdf = perm_gdf.set_crs("EPSG:4326")
    max_gdf  = max_gdf.set_crs("EPSG:4326")

    # --------------------------------------------------
    # Merge polygons & remove holes
    # --------------------------------------------------
    perm_gdf = merge_and_fill(perm_gdf)
    max_gdf  = merge_and_fill(max_gdf)

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))

        max_gdf.plot(ax=ax, color="lightblue", edgecolor="blue", alpha=0.5)
        perm_gdf.plot(ax=ax, facecolor="none", edgecolor="navy", linewidth=1.2)

        legend_elements = [
            Patch(facecolor="lightblue", edgecolor="black", label="Maximum water extent"),
            Patch(facecolor="none", edgecolor="navy", label="Permanent water"),
        ]

        ax.legend(handles=legend_elements, fontsize=9, loc="best")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("GSW reservoir water masks")
        ax.grid(alpha=0.3)

        plt.show()

    return perm_gdf, max_gdf
