import os
import ee
import geopandas as gpd

from .water_masks import water_masks
from .classification import run_full_classification
from .generate_time_series import generate_altimetry_timeseries


def run_reservoir_analysis(
    aoi_gdf: gpd.GeoDataFrame,
    altimetry_folder: str,
    output_csv: str,
    occurrence_threshold: int = 90,
):
    """
    End-to-end reservoir analysis pipeline.

    Parameters
    ----------
    aoi_gdf : GeoDataFrame
        Area of interest (polygon or multipolygon, EPSG:4326)
    altimetry_folder : str
        Folder containing altimetry NetCDF files
    output_csv : str
        Output CSV path
    occurrence_threshold : int
        GSW occurrence threshold for permanent water
    """

    # --------------------------------------------------
    # Earth Engine initialization
    # --------------------------------------------------
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

    # --------------------------------------------------
    # Sanity checks
    # --------------------------------------------------
    if aoi_gdf.empty:
        raise ValueError("AOI GeoDataFrame is empty")

    if aoi_gdf.crs is None:
        raise ValueError("AOI GeoDataFrame has no CRS")

    if aoi_gdf.crs.to_string() != "EPSG:4326":
        aoi_gdf = aoi_gdf.to_crs("EPSG:4326")

    # --------------------------------------------------
    # Output directory
    # --------------------------------------------------
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # --------------------------------------------------
    # STEP 1: Water masks (AOI-BASED)
    # --------------------------------------------------
    perm_gdf, max_gdf = water_masks(
        aoi_gdf=aoi_gdf,
        occurrence_threshold=occurrence_threshold,
        plot=False,
    )

    if perm_gdf.empty or max_gdf.empty:
        raise RuntimeError("Water mask extraction failed (empty result)")

    # --------------------------------------------------
    # STEP 2: Reservoir classification
    # --------------------------------------------------
    reservoir_class, class_metrics = run_full_classification(
        base_folder=altimetry_folder,
        max_water_input=max_gdf,
        perm_water_input=perm_gdf,
    )

    # --------------------------------------------------
    # STEP 3: Generate altimetry time series
    # --------------------------------------------------
    ts_df = generate_altimetry_timeseries(
        nc_folder=altimetry_folder,
        max_gdf=max_gdf,
        reservoir_class=reservoir_class,
        output_csv=output_csv,
    )

    # --------------------------------------------------
    # Return results
    # --------------------------------------------------
    return {
        "reservoir_class": reservoir_class,
        "classification_metrics": class_metrics,
        "output_csv": output_csv,
        "timeseries": ts_df,
        "perm_gdf": perm_gdf,
        "max_gdf": max_gdf,
        "aoi_gdf": aoi_gdf,
    }
