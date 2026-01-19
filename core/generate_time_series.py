def generate_altimetry_timeseries(
    nc_folder,
    max_gdf,
    reservoir_class,
    output_csv,
    s1_search_days=90,
):
    """
    Generate reservoir water level time series.
    """

    # --------------------------------------------------
    # Imports
    # --------------------------------------------------
    import os, glob
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import shapely
    import ee, geemap

    from .altimetry_extractors import extract_altimetry_data

    # --------------------------------------------------
    # CLASS PARSER
    # --------------------------------------------------
    def parse_reservoir_class(res_class):
        if res_class is None:
            raise ValueError("reservoir_class cannot be None")
        if not isinstance(res_class, str) or len(res_class) != 2:
            raise ValueError(f"Invalid reservoir_class: {res_class}")
        return int(res_class[0]), res_class[1]

    original_class = reservoir_class
    base_class, terrain_flag = parse_reservoir_class(reservoir_class)

    degraded = False
    use_s1 = base_class in [3, 4]
    buffer_m = -1000 if base_class in [1, 2] else 0

    # --------------------------------------------------
    # GEOMETRY HELPERS
    # --------------------------------------------------
    def clean_gdf(gdf):
        gdf = gdf[gdf.geometry.notnull()]
        gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf[gdf.is_valid & ~gdf.is_empty]
        return gdf

    def safe_union_geoms(geoms):
        geoms = [g.buffer(0) for g in geoms if g and g.is_valid and not g.is_empty]
        if not geoms:
            return None
        try:
            return shapely.union_all(geoms)
        except Exception:
            return None

    def buffer_gdf_safe(gdf, buffer_m):
        g = gdf.to_crs(3857)
        g["geometry"] = g.geometry.buffer(buffer_m).buffer(0)
        g = g[g.is_valid & ~g.is_empty]
        return g.to_crs(4326)

    # --------------------------------------------------
    # FILTER HELPERS
    # --------------------------------------------------
    def mad_outlier_removal(df, column, threshold=2.0):
        if df.empty:
            return df
        med = df[column].median()
        mad = np.median(np.abs(df[column] - med))
        if mad == 0:
            return df
        z = 0.6745 * (df[column] - med) / mad
        return df[np.abs(z) <= threshold]

    def rolling_iqr_outlier_removal(df, column, time_window="90D", threshold_multiplier=1.0):
        if df.empty:
            return df
        Q1 = df[column].rolling(window=time_window, center=True, min_periods=3).quantile(0.25)
        Q3 = df[column].rolling(window=time_window, center=True, min_periods=3).quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold_multiplier * IQR
        upper = Q3 + threshold_multiplier * IQR
        return df[(df[column] >= lower) & (df[column] <= upper)]

    # --------------------------------------------------
    # MEDIAN POINT PER PASS
    # --------------------------------------------------
    def compute_median_point(df):
        if df.empty or "range" not in df.columns:
            return None

        df = df.copy()

        for col in ["iono", "dry", "wet", "pole", "solid", "geoid", "load_tide"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0)

        df["range"] = df["range"].fillna(0.0)

        mission = str(df["mission"].iloc[0])
        is_s3 = "S3" in mission or "Sentinel" in mission

        if is_s3:
            elev = (
                df["altitude"]
                - (
                    df["range"]
                    + df["dry"] + df["wet"]
                    + df["pole"] + df["solid"]
                    + df["iono"] + df["load_tide"]
                )
                - df["geoid"]
            )
        else:
            elev = (
                df["altitude"]
                - (
                    df["range"]
                    + df["dry"] + df["wet"]
                    + df["pole"] + df["solid"]
                    + df["iono"]
                )
                - df["geoid"]
            )

        df["elevation"] = elev
        df = df.dropna(subset=["elevation", "latitude", "longitude", "date"])
        if df.empty:
            return None

        med = df["elevation"].median()
        idx = (df["elevation"] - med).abs().idxmin()
        pt = df.loc[idx]

        return {
            "mission": pt["mission"],
            "date": pd.to_datetime(pt["date"]),
            "elevation": med,
            "latitude": pt["latitude"],
            "longitude": pt["longitude"],
        }

    # --------------------------------------------------
    # INIT EE
    # --------------------------------------------------
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

    # --------------------------------------------------
    # GEOMETRY PREPARATION WITH DEGRADATION
    # --------------------------------------------------
    max_gdf = clean_gdf(max_gdf)
    buffered_max = buffer_gdf_safe(max_gdf, buffer_m)
    max_union = safe_union_geoms(buffered_max.geometry)

    if max_union is None or max_union.is_empty:
        if buffer_m < 0:
            print("[INFO] No geometry after negative buffer — degrading to class 3/4")
            degraded = True
            buffer_m = 0
            use_s1 = True

            buffered_max = buffer_gdf_safe(max_gdf, buffer_m)
            max_union = safe_union_geoms(buffered_max.geometry)

        if max_union is None or max_union.is_empty:
            print("[WARN] Geometry invalid even after degradation — skipping reservoir")
            return None

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------
    records = []

    for nc in sorted(glob.glob(os.path.join(nc_folder, "**/*.nc"), recursive=True)):
        try:
            df = extract_altimetry_data(nc)
        except Exception:
            continue

        required = {"latitude", "longitude", "date", "mission", "altitude"}
        if not required.issubset(df.columns):
            continue

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326",
        )

        gdf_in = gdf[gdf.geometry.within(max_union)]

        if gdf_in.empty and buffer_m < 0:
            print("[INFO] Track missed buffered geometry — degrading to class 3/4")
            degraded = True

            buffered_max_fb = buffer_gdf_safe(max_gdf, 0)
            max_union_fb = safe_union_geoms(buffered_max_fb.geometry)
            if max_union_fb is None or max_union_fb.is_empty:
                continue

            gdf_in = gdf[gdf.geometry.within(max_union_fb)]

        if gdf_in.empty:
            continue

        rep = compute_median_point(gdf_in.drop(columns="geometry"))
        if rep:
            records.append(rep)

    if not records:
        return None

    # --------------------------------------------------
    # FINAL FILTERING
    # --------------------------------------------------
    ts = (
        pd.DataFrame(records)
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
    )

    ts = mad_outlier_removal(ts, "elevation", 2.0)
    ts = rolling_iqr_outlier_removal(ts, "elevation")

    ts = ts.reset_index()
    ts.to_csv(output_csv, index=False, float_format="%.3f")

    # --------------------------------------------------
    # FINAL CLASS UPDATE
    # --------------------------------------------------
    final_class = reservoir_class
    if degraded:
        final_class = f"3{terrain_flag}"

    return {
        "timeseries": ts,
        "reservoir_class": final_class,
        "original_class": original_class,
        "degraded": degraded,
    }
