def generate_altimetry_timeseries(
    nc_folder,
    max_gdf,
    reservoir_class,
    output_dir,
    s1_search_days=90,
):
    """
    Generate a reservoir water-level time series using:

    Median elevation of all valid altimetry points per pass
    (with representative latitude/longitude closest to the median)

    Filtering:
      - Global MAD filter (threshold = 2.0)
      - Rolling time-based IQR filter (90D, multiplier = 1.0)
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
    # Output path
    # --------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    MEDIAN_CSV = os.path.join(output_dir, "MED.csv")

    # --------------------------------------------------
    # CLASS PARSER
    # --------------------------------------------------
    def parse_reservoir_class(res_class):
        if not isinstance(res_class, str) or len(res_class) != 2:
            raise ValueError(f"Invalid reservoir_class: {res_class}")
        return int(res_class[0]), res_class[1]

    base_class, _ = parse_reservoir_class(reservoir_class)
    use_s1 = base_class in [3, 4]

    # −1 km inward buffer ONLY for Class 1–2
    buffer_m = -1000 if base_class in [1, 2] else 0

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------
    def buffer_gdf(gdf, buffer_m):
        g = gdf.to_crs(3857)
        g["geometry"] = g.geometry.buffer(buffer_m)
        g = g[~g.is_empty]
        return g.to_crs(4326)

    def mad_filter(df, col, thr=2.0):
        if df.empty:
            return df
        med = df[col].median()
        mad = np.median(np.abs(df[col] - med))
        if mad == 0:
            return df
        z = 0.6745 * (df[col] - med) / mad
        return df[np.abs(z) <= thr]

    def rolling_iqr_filter(df, col, window="90D", mult=1.0):
        if df.empty:
            return df
        q1 = df[col].rolling(window, center=True, min_periods=3).quantile(0.25)
        q3 = df[col].rolling(window, center=True, min_periods=3).quantile(0.75)
        iqr = q3 - q1
        return df[(df[col] >= q1 - mult * iqr) & (df[col] <= q3 + mult * iqr)]

    # --------------------------------------------------
    # SENTINEL-1 MASK
    # --------------------------------------------------
    def sentinel1_mask(region_ee, target_date):
        coll = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region_ee)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .select("VV")
        )

        coll = coll.filterDate(
            ee.Date(target_date - timedelta(days=s1_search_days)),
            ee.Date(target_date + timedelta(days=s1_search_days)),
        )

        if coll.size().getInfo() == 0:
            return None

        imgs = coll.toList(coll.size()).getInfo()
        times = np.array([i["properties"]["system:time_start"] for i in imgs])
        idx = np.argmin(np.abs(times - target_date.timestamp() * 1000))
        nearest = datetime.fromtimestamp(times[idx] / 1000, tz=UTC)

        img = (
            coll.filterDate(
                nearest.strftime("%Y-%m-%d"),
                (nearest + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            .mosaic()
            .focal_median(1)
        )

        try:
            sample = img.sample(region=region_ee, scale=30, numPixels=500)
            arr = np.array(sample.aggregate_array("VV").getInfo())
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return None
            return img.lt(threshold_otsu(arr))
        except Exception:
            return None

    # --------------------------------------------------
    # INIT EE
    # --------------------------------------------------
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

    # --------------------------------------------------
    # BUFFER MAX EXTENT
    # --------------------------------------------------
    buffered_max = buffer_gdf(max_gdf, buffer_m)
    max_union = buffered_max.geometry.union_all()
    region_ee = geemap.geopandas_to_ee(buffered_max)

    # --------------------------------------------------
    # OUTPUT CONTAINER
    # --------------------------------------------------
    ts_median = []

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------
    for nc in sorted(glob.glob(os.path.join(nc_folder, "**/*.nc"), recursive=True)):
        try:
            df = extract_altimetry_data(nc)
        except Exception:
            continue

        required = {"latitude", "longitude", "date", "mission", "altitude", "range"}
        if not required.issubset(df.columns):
            continue

        # Fill missing corrections
        for c in ["iono", "dry", "wet", "pole", "solid"]:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = df[c].fillna(0.0)

        if "geoid" not in df.columns:
            df["geoid"] = 0.0
        df["geoid"] = df["geoid"].fillna(0.0)

        # Correct elevation
        df["elevation"] = (
            df["altitude"]
            - (df["range"] + df["dry"] + df["wet"]
               + df["pole"] + df["solid"] + df["iono"])
            - df["geoid"]
        )

        df = df.dropna(subset=["elevation", "latitude", "longitude", "date"])

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326",
        )

        gdf = gdf[gdf.geometry.within(max_union)]
        if gdf.empty:
            continue

        # Sentinel-1 water mask (Class 3–4 only)
        if use_s1:
            mask = sentinel1_mask(region_ee, pd.to_datetime(gdf["date"].iloc[0]))
            if mask is None:
                continue

            fc = mask.sampleRegions(
                geemap.geopandas_to_ee(gdf),
                scale=30,
                geometries=True,
            )

            if fc.size().getInfo() == 0:
                continue

            gdf = geemap.ee_to_gdf(fc)
            band = next(
                c for c in gdf.columns
                if c not in ("geometry", "id", "system:index")
            )
            gdf = gdf[gdf[band].astype(bool)]

            if gdf.empty:
                continue

        date = pd.to_datetime(gdf["date"].iloc[0])
        mission = gdf["mission"].iloc[0]

        # ---------------- Median time series ----------------
        med = gdf["elevation"].median()
        idx = (gdf["elevation"] - med).abs().idxmin()
        pt = gdf.loc[idx]

        ts_median.append({
            "mission": mission,
            "date": date,
            "elevation": med,
            "latitude": pt["latitude"],
            "longitude": pt["longitude"],
        })

    # --------------------------------------------------
    # SAVE OUTPUT WITH FILTERING
    # --------------------------------------------------
    if ts_median:
        ts = (
            pd.DataFrame(ts_median)
            .drop_duplicates("date")
            .sort_values("date")
            .set_index("date")
        )

        ts = mad_filter(ts, "elevation")
        ts = rolling_iqr_filter(ts, "elevation")

        ts.reset_index().to_csv(
            MEDIAN_CSV, index=False, float_format="%.3f"
        )

    return MEDIAN_CSV
