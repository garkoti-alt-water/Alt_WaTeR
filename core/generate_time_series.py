def generate_altimetry_timeseries(
    nc_folder,
    max_gdf,
    reservoir_class,
    output_csv,
    class_color=None,
    s1_search_days=90,
    make_plots=True,
):
    """
    Generate reservoir water level time series using
    Sentinel-1 VV/IW + nearest-date + Otsu water masking.
    """

    # --------------------------------------------------
    # Imports (local)
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
    # OUTLIER FILTER UTILITIES
    # --------------------------------------------------
    def mad_outlier_removal(df, column, threshold=3.5):
        if df.empty:
            return df
        data = df[column]
        median = data.median()
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return df
        modified_z = 0.6745 * (data - median) / mad
        return df[np.abs(modified_z) <= threshold]

    def rolling_iqr_outlier_removal(df, column, window="90D", k=1.0):
        if df.empty:
            return df
        Q1 = df[column].rolling(window, center=True, min_periods=3).quantile(0.25)
        Q3 = df[column].rolling(window, center=True, min_periods=3).quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[column] >= Q1 - k * IQR) & (df[column] <= Q3 + k * IQR)]

    # --------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------
    def buffer_gdf(gdf, buffer_m):
        g = gdf.to_crs(3857)
        g["geometry"] = g.geometry.buffer(buffer_m)
        g = g[~g.is_empty]
        return g.to_crs(4326)

    def compute_median_point(df):
        if df.empty:
            return None

        for col in ["iono","dry","wet","pole","solid","geoid","load_tide"]:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0)

        is_s3 = "S3" in str(df["mission"].iloc[0])

        elev = np.where(
            is_s3,
            df["altitude"] - (
                df["range"] + df["dry"] + df["wet"] +
                df["pole"] + df["solid"] +
                df["iono"] + df["load_tide"]
            ) - df["geoid"],
            df["altitude"] - (
                df["range"] + df["dry"] + df["wet"] +
                df["pole"] + df["solid"] +
                df["iono"]
            ) - df["geoid"]
        )

        df = df.copy()
        df["elevation"] = elev
        df = df.dropna(subset=["elevation","latitude","longitude","date"])
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
    # SENTINEL-1 NEAREST-DATE + OTSU MASK
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
        times = [i["properties"]["system:time_start"] for i in imgs]
        idx = int(np.argmin([
            abs(t - target_date.timestamp() * 1000) for t in times
        ]))

        nearest = datetime.fromtimestamp(times[idx] / 1000, tz=UTC)

        img = (
            coll
            .filterDate(
                nearest.strftime("%Y-%m-%d"),
                (nearest + timedelta(days=1)).strftime("%Y-%m-%d")
            )
            .mosaic()
            .focal_median(1)
        )

        arr = np.array(
            img.sampleRectangle(region=region_ee).get("VV").getInfo()
        )
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return None

        thr = threshold_otsu(arr)
        return img.lt(thr)

    # --------------------------------------------------
    # INITIALIZE EE
    # --------------------------------------------------
    ee.Initialize()

    # --------------------------------------------------
    # CLASS LOGIC
    # --------------------------------------------------
    buffer_m = -300 if reservoir_class in [1, 2] else 0
    use_s1 = reservoir_class not in [1, 2]

    buffered_max = buffer_gdf(max_gdf, buffer_m)
    max_union = buffered_max.geometry.union_all()
    region_ee = geemap.geopandas_to_ee(buffered_max)

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------
    results = []

    for nc in sorted(glob.glob(os.path.join(nc_folder, "**/*.nc"), recursive=True)):
        try:
            df = extract_altimetry_data(nc)
        except Exception:
            continue

        if not {"latitude","longitude","date"}.issubset(df.columns):
            continue

        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )

        gdf = gdf[gdf.geometry.within(max_union)]
        if gdf.empty:
            continue

        if not use_s1:
            rep = compute_median_point(gdf.drop(columns="geometry"))
        else:
            mask = sentinel1_mask(region_ee, pd.to_datetime(df["date"].iloc[0]))
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

            # 🔑 KEEP ONLY WATER POINTS
            band = next(
                (c for c in sampled.columns if c not in ("geometry","system:index","id")),
                None
            )
            if band is None:
                continue

            water = sampled[sampled[band].astype(bool)]
            if water.empty:
                continue

            rep = compute_median_point(water.drop(columns="geometry"))

        if rep:
            results.append(rep)

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------
    if not results:
        raise RuntimeError("No valid time-series points produced.")

    ts = pd.DataFrame(results).sort_values("date").drop_duplicates("date")

    ts = mad_outlier_removal(ts, "elevation")
    ts = rolling_iqr_outlier_removal(ts, "elevation", window=5)


    ts.to_csv(output_csv, index=False, float_format="%.3f")
    return ts
