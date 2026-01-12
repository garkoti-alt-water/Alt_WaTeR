def generate_altimetry_timeseries(
    nc_folder,
    max_gdf,
    reservoir_class,
    output_csv,
    s1_search_days=90,
):
    """
    Generate reservoir water level time series using:
    - Direct altimetry for Classes 1A/1B/2A/2B
    - Sentinel-1 VV/IW + nearest-date + Otsu masking for Classes 3A–4B

    Logic:
      1. Filter points inside maximum water extent
         - Class 1–2 → negative buffer (−300 m)
         - Class 3–4 → no buffer
      2. If Class 3–4 → apply Sentinel-1 masking
      3. Compute per-pass median elevation
      4. Apply global MAD filter (threshold = 2.0)
      5. Apply rolling time-based IQR filter (90D, multiplier = 1.0)
    """

    # --------------------------------------------------
    # Imports
    # --------------------------------------------------
    import os, glob
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    import ee, geemap
    from datetime import datetime, timedelta, UTC
    from skimage.filters import threshold_otsu

    from .altimetry_extractors import extract_altimetry_data

    # --------------------------------------------------
    # CLASS PARSER (STRICT)
    # --------------------------------------------------
    def parse_reservoir_class(res_class):
        if res_class is None:
            raise ValueError("reservoir_class cannot be None")
        if not isinstance(res_class, str) or len(res_class) != 2:
            raise ValueError(f"Invalid reservoir_class: {res_class}")
        return int(res_class[0]), res_class[1]

    base_class, terrain_flag = parse_reservoir_class(reservoir_class)

    use_s1 = base_class in [3, 4]
    buffer_m = -300 if base_class in [1, 2] else 0

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def buffer_gdf(gdf, buffer_m):
        g = gdf.to_crs(3857)
        g["geometry"] = g.geometry.buffer(buffer_m)
        g = g[~g.is_empty]
        return g.to_crs(4326)

    # -------- GLOBAL MAD FILTER --------
    def mad_outlier_removal(df, column, threshold=2.0):
        if df.empty:
            return df
        med = df[column].median()
        mad = np.median(np.abs(df[column] - med))
        if mad == 0:
            return df
        z = 0.6745 * (df[column] - med) / mad
        return df[np.abs(z) <= threshold]

    # -------- TIME-BASED ROLLING IQR --------
    def rolling_iqr_outlier_removal(
        df,
        column,
        time_window="90D",
        threshold_multiplier=1.0,
    ):
        if df.empty:
            return df

        Q1 = df[column].rolling(
            window=time_window, center=True, min_periods=3
        ).quantile(0.25)

        Q3 = df[column].rolling(
            window=time_window, center=True, min_periods=3
        ).quantile(0.75)

        IQR = Q3 - Q1
        lower = Q1 - threshold_multiplier * IQR
        upper = Q3 + threshold_multiplier * IQR

        mask = (df[column] >= lower) & (df[column] <= upper)
        return df[mask]

    # --------------------------------------------------
    # MEDIAN POINT COMPUTATION (PER PASS)
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
                    + df["dry"] + df["wet"] + df["pole"]
                    + df["solid"] + df["iono"] + df["load_tide"]
                )
                - df["geoid"]
            )
        else:
            elev = (
                df["altitude"]
                - (
                    df["range"]
                    + df["dry"] + df["wet"]
                    + df["pole"] + df["solid"] + df["iono"]
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
    # SENTINEL-1 MASK
    # --------------------------------------------------
    def sentinel1_mask(region_ee, target_date):
        coll = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region_ee)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            ))
            .select("VV")
        )

        coll = coll.filterDate(
            ee.Date(target_date - timedelta(days=s1_search_days)),
            ee.Date(target_date + timedelta(days=s1_search_days))
        )

        if coll.size().getInfo() == 0:
            return None

        imgs = coll.toList(coll.size()).getInfo()
        times = np.array([i["properties"]["system:time_start"] for i in imgs])
        idx = np.argmin(np.abs(times - target_date.timestamp() * 1000))
        nearest = datetime.fromtimestamp(times[idx] / 1000, tz=UTC)

        img = (
            coll
            .filterDate(nearest.strftime("%Y-%m-%d"),
                        (nearest + timedelta(days=1)).strftime("%Y-%m-%d"))
            .mosaic()
            .focal_median(1)
        )

        arr = np.array(img.sampleRectangle(region=region_ee).get("VV").getInfo())
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None

        return img.lt(threshold_otsu(arr))

    # --------------------------------------------------
    # INITIALIZE EE
    # --------------------------------------------------
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

    buffered_max = buffer_gdf(max_gdf, buffer_m)
    max_union = buffered_max.geometry.union_all()
    region_ee = geemap.geopandas_to_ee(buffered_max)

    # --------------------------------------------------
    # MAIN LOOP (PER PASS)
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
            crs="EPSG:4326"
        )

        # --- STEP 1: Max-extent filtering ---
        gdf = gdf[gdf.geometry.within(max_union)]
        if gdf.empty:
            continue

        # --- STEP 2: Sentinel-1 masking (Class 3–4 only) ---
        if use_s1:
            mask = sentinel1_mask(
                region_ee,
                pd.to_datetime(gdf["date"].iloc[0])
            )
            if mask is None:
                continue

            fc = mask.sampleRegions(
                geemap.geopandas_to_ee(gdf),
                scale=30,
                geometries=True
            )
            if fc.size().getInfo() == 0:
                continue

            sampled = geemap.ee_to_gdf(fc)
            band = next(
                (c for c in sampled.columns
                 if c not in ("geometry", "system:index", "id")),
                None
            )
            if band is None:
                continue

            gdf = sampled[sampled[band].astype(bool)]
            if gdf.empty:
                continue

        # --- STEP 3: Per-pass median ---
        rep = compute_median_point(gdf.drop(columns="geometry"))
        if rep:
            records.append(rep)

    if not records:
        raise RuntimeError("No valid time-series points produced.")

    # --------------------------------------------------
    # FINAL TIME SERIES + FILTERING
    # --------------------------------------------------
    ts = (
        pd.DataFrame(records)
        .drop_duplicates("date")
        .sort_values("date")
        .set_index("date")
    )

    # --- Global MAD ---
    ts = mad_outlier_removal(ts, column="elevation", threshold=2.0)

    # --- Rolling IQR (time-based) ---
    ts = rolling_iqr_outlier_removal(
        ts,
        column="elevation",
        time_window="90D",
        threshold_multiplier=1.0
    )

    ts = ts.reset_index()
    ts.to_csv(output_csv, index=False, float_format="%.3f")
    return ts
