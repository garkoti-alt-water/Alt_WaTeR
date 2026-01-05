import os
import ee

from .water_masks import water_masks
from .classification import run_full_classification
from .generate_time_series import generate_altimetry_timeseries


def run_reservoir_analysis(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    altimetry_folder: str,
    output_csv: str,
    occurrence_threshold: int = 90,
):
    """
    End-to-end reservoir analysis pipeline.

    Returns
    -------
    dict with:
        - reservoir_class : int
        - output_csv : str
        - timeseries : pandas.DataFrame
        - map_fig : matplotlib.figure.Figure
        - ts_fig : matplotlib.figure.Figure
        - perm_gdf : geopandas.GeoDataFrame
        - max_gdf : geopandas.GeoDataFrame
    """

    # --------------------------------------------------
    # Safety checks
    # --------------------------------------------------
    if not ee.data._credentials:
        raise RuntimeError(
            "Earth Engine not initialized. Run ee.Authenticate() and ee.Initialize() first."
        )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # --------------------------------------------------
    # STEP 1: Water masks
    # --------------------------------------------------
    perm_gdf, max_gdf = water_masks(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        occurrence_threshold=occurrence_threshold,
        plot=False,
    )

    # --------------------------------------------------
    # STEP 2: Reservoir classification
    # --------------------------------------------------
    reservoir_class, _ = run_full_classification(
        base_folder=altimetry_folder,
        max_water_input=max_gdf,
        perm_water_input=perm_gdf,
    )

    # --------------------------------------------------
    # STEP 3: Generate time series + figures
    # --------------------------------------------------
    ts_df = generate_altimetry_timeseries(
        nc_folder=altimetry_folder,
        max_gdf=max_gdf,
        reservoir_class=reservoir_class,
        output_csv=output_csv
    )


    # --------------------------------------------------
    # Return EVERYTHING needed by GUI
    # --------------------------------------------------
    return {
        "reservoir_class": reservoir_class,
        "output_csv": output_csv,
        "timeseries": ts_df,
        "perm_gdf": perm_gdf,
        "max_gdf": max_gdf,
    }

