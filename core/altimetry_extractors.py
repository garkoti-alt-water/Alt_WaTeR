from shapely.geometry import Polygon
import os
import glob
import warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from netCDF4 import Dataset
from scipy.interpolate import interp1d
import ee
import geemap

import zipfile
import tempfile
import shutil


# ----------------- HELPERS -----------------
def interpolate_1hz_to_20hz(data_1hz, time_1hz, time_20hz):
    """Interpolate 1Hz -> 20Hz safely (handles masked arrays & NaNs)."""
    data_1hz = np.ma.filled(data_1hz, np.nan)
    time_1hz = np.ma.filled(time_1hz, np.nan)
    time_20hz = np.ma.filled(time_20hz, np.nan)
    mask = ~np.isnan(data_1hz) & ~np.isnan(time_1hz)
    if np.count_nonzero(mask) < 2:
        return np.full_like(time_20hz, np.nan, dtype=float)
    f = interp1d(time_1hz[mask], data_1hz[mask], kind="linear", bounds_error=False, fill_value="extrapolate")
    return f(time_20hz)

def extract_s3_zip(zip_path):
    """
    Extract a Sentinel-3 .SEN3.zip to a temporary directory
    and return path to enhanced_measurement.nc
    """
    
    tmpdir = tempfile.mkdtemp(prefix="s3_")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmpdir)

    
    sen3_dirs = glob.glob(os.path.join(tmpdir, "*.SEN3"))
    if not sen3_dirs:
        shutil.rmtree(tmpdir)
        raise RuntimeError(f"[S3] ERROR: No .SEN3 directory found inside ZIP")

    sen3_dir = sen3_dirs[0]
    

    nc_path = os.path.join(sen3_dir, "enhanced_measurement.nc")
    if not os.path.exists(nc_path):
       
        
        shutil.rmtree(tmpdir)
        raise RuntimeError("[S3] ERROR: enhanced_measurement.nc not found")

   
    return nc_path, tmpdir

def try_vars(ds, candidates):
    """Return first existing variable array from candidates or raise KeyError."""
    for name in candidates:
        if name in ds.variables:
            return ds.variables[name][:]
    raise KeyError(f"None of candidate variable names found: {candidates}")

def discover_altimetry_files(base_folder):
    files = []

    # Sentinel-3 ZIP
    files.extend(
        glob.glob(os.path.join(base_folder, "**/*.SEN3.zip"), recursive=True)
    )

    # Sentinel-3 extracted
    files.extend(
        glob.glob(
            os.path.join(base_folder, "**/*.SEN3/enhanced_measurement.nc"),
            recursive=True,
        )
    )

    # Other missions
    for f in glob.glob(os.path.join(base_folder, "**/*.nc"), recursive=True):
        fl = f.lower()
        if ".sen3/" in fl:
            continue
        if any(x in fl for x in ["standard_measurement", "calibration", "aux", "noise"]):
            continue
        files.append(f)

    return sorted(set(files))


# ----------------- EXTRACTORS -----------------
def extract_sentinel3_data(nc_path):
    ds = Dataset(nc_path, "r")
    fname = os.path.basename(nc_path)
    parent = os.path.basename(os.path.dirname(nc_path)).upper()

    mission = (
        "S3A" if "S3A" in parent or "S3A" in fname.upper()
        else "S3B" if "S3B" in parent or "S3B" in fname.upper()
        else "S3"
    )

    file_date = pd.NaT
    if hasattr(ds, "equator_time"):
        try:
            file_date = datetime.strptime(ds.equator_time.split("T")[0], "%Y-%m-%d")
        except Exception:
            pass

    lats = try_vars(ds, ["lat_20_ku", "latitude", "lat"])
    lons_raw = try_vars(ds, ["lon_20_ku", "longitude", "lon"])
    lons = np.where(lons_raw > 180, lons_raw - 360, lons_raw)
    tims = try_vars(ds, ["time_20_ku", "time_20", "time"])
    rng = try_vars(ds, ["range_ocog_20_ku", "range_ocog", "range"])
    alt = try_vars(ds, ["alt_20_ku", "altitude", "alt"])

    def get_1hz(names):
        try:
            return try_vars(ds, names)
        except KeyError:
            return None

    t1 = get_1hz(["time_01", "time_1hz"])
    pole = interpolate_1hz_to_20hz(get_1hz(["pole_tide"]), t1, tims) if t1 is not None else np.nan
    solid = interpolate_1hz_to_20hz(get_1hz(["solid_earth_tide"]), t1, tims) if t1 is not None else np.nan
    iono = interpolate_1hz_to_20hz(get_1hz(["iono_cor_gim"]), t1, tims) if t1 is not None else np.nan
    geoid = interpolate_1hz_to_20hz(get_1hz(["geoid"]), t1, tims) if t1 is not None else np.nan

    ds.close()

    df = pd.DataFrame({
        "mission": mission,
        "date": pd.to_datetime(file_date),
        "latitude": lats,
        "longitude": lons,
        "time_20hz": tims,
        "altitude": alt,
        "range": rng,
        "pole": pole,
        "solid": solid,
        "iono": iono,
        "geoid": geoid,
    })

    return df


def extract_jason3_or_s6_data(nc_path):
    """Jason-3 / Sentinel-6 extractor (typical structure)."""
    ds = Dataset(nc_path, "r")
    fname = os.path.basename(nc_path)
    mission = fname.split("_")[0] if "_" in fname else "JA3"

    # --- UPDATED DATE LOGIC ---
    file_date = pd.NaT # Default to Not-a-Time
    try:
        if mission == "JA3":
            date_str = fname.split("_")[4]
        else:
            parts = fname.split("_")
            date_str = parts[10] if len(parts) > 10 else parts[-1]
        file_date = datetime.strptime(date_str[:8], "%Y%m%d")
    except Exception:
        print(f"Warning: Jason/S6 file {fname} - could not parse date from filename.")
        pass # file_date remains pd.NaT
    # --- END UPDATED DATE LOGIC ---

    try:
        g20 = ds.groups["data_20"]
        g01 = ds.groups["data_01"]
    except Exception as e:
        ds.close()
        raise RuntimeError(f"Jason/S6 structure not found: {e}")

    if mission == "JA3":
        lats = g20.variables["latitude"][:]
        lons_raw = g20.variables["longitude"][:]
        lons = np.where(lons_raw > 180, lons_raw - 360, lons_raw)
        t20 = g20.variables["time"][:]
        rng = g20.groups["ku"].variables["range_ocog"][:]
        alt = g20.variables["altitude"][:]
        wet = g20.variables["model_wet_tropo_cor_measurement_altitude"][:]
        dry = g20.variables["model_dry_tropo_cor_measurement_altitude"][:]
        t1 = g01.variables["time"][:]
        pole = interpolate_1hz_to_20hz(g01.variables["pole_tide"][:], t1, t20)
        solid = interpolate_1hz_to_20hz(g01.variables["solid_earth_tide"][:], t1, t20)
        iono = interpolate_1hz_to_20hz(g01.groups["ku"].variables["iono_cor_gim"][:], t1, t20)
        geoid = interpolate_1hz_to_20hz(g01.variables["geoid"][:], t1, t20)

        df = pd.DataFrame({
        "mission": mission,
        "date": file_date,
        "latitude": lats,
        "longitude": lons,
        "time_20hz": t20,
        "altitude": alt,
        "range": rng,
        "dry": dry,
        "wet": wet,
        "pole": pole,
        "solid": solid,
        "iono": iono,
        "geoid": geoid,
    })

    else:
        lats = ds.groups['data_20'].groups['ku'].variables['latitude'][:]
        lons = ds.groups['data_20'].groups['ku'].variables['longitude'][:]
        lons = np.where(lons > 180, lons - 360, lons)
        tims = ds.groups['data_20'].groups['ku'].variables['time'][:]
        range_counter = ds.groups['data_20'].groups['ku'].variables['range_ocog'][:]
        geoid = ds.groups['data_20'].groups['ku'].variables['geoid'][:]
        altitude= ds.groups['data_20'].groups['ku'].variables['altitude'][:]
        wet_tropo= ds.groups['data_20'].groups['ku'].variables['model_wet_tropo_cor_measurement_altitude'][:]
        dry_tropo= ds.groups['data_20'].groups['ku'].variables['model_dry_tropo_cor_measurement_altitude'][:]
        pole_tide_1hz = ds.groups['data_01'].variables['pole_tide'][:]
        timestamps_1hz = ds.groups['data_01'].variables['time'][:]
        solid_tide_1hz = ds.groups['data_01'].variables['solid_earth_tide'][:] 
        iono_1hz=ds.groups['data_01']['ku'].variables['iono_cor_gim'][:] 
        
        
            
           # Interpolate 1Hz data to match 20Hz timestamps
        pole_tide_20hz = interpolate_1hz_to_20hz(pole_tide_1hz, timestamps_1hz, tims)
        solid_tide_20hz = interpolate_1hz_to_20hz(solid_tide_1hz, timestamps_1hz, tims)
        iono_20hz=interpolate_1hz_to_20hz(iono_1hz, timestamps_1hz, tims)

        df = pd.DataFrame({
        "mission": mission,
        "date": file_date,
        "latitude": lats,
        "longitude": lons,
        "time_20hz": tims,
        "altitude": altitude,
        "range": range_counter,
        "dry": dry_tropo,
        "wet": wet_tropo,
        "pole": pole_tide_20hz,
        "solid": solid_tide_20hz,
        "iono": iono_20hz,
        "geoid": geoid,
    })

    ds.close()

    
    df['date'] = pd.to_datetime(df['date']) # Ensure column is datetime type
    return df

# --- THIS FUNCTION IS NOW REPLACED ---
def extract_swot_data(nc_path):
    """SWOT extractor based on user-provided Jason-3-style structure."""
    
    # Date logic from user snippet
    file_date = pd.NaT # Default
    try:
        date_str = os.path.basename(nc_path).split('_')[4]
        year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
        file_date = datetime(year, month, day)
    except Exception:
        # Fallback to the logic from the script's other extractors
        print(f"Warning: SWOT file {os.path.basename(nc_path)} - could not parse date from split part. Trying standard parse.")
        try:
            parts = os.path.basename(nc_path).split("_")
            date_token = next((p for p in parts if p.isdigit() and len(p) == 8), None)
            if date_token:
                file_date = datetime.strptime(date_token, "%Y%m%d")
            else:
                 print(f"Warning: SWOT file {os.path.basename(nc_path)} - no 8-digit date token found in filename.")
        except Exception:
             print(f"Warning: SWOT file {os.path.basename(nc_path)} - could not parse date from filename.")
             pass # file_date remains pd.NaT

    ds = Dataset(nc_path, 'r')

    try:
        lats = ds.groups['data_20'].variables['latitude'][:]
        lons_raw = ds.groups['data_20'].variables['longitude'][:]
        lons = np.where(lons_raw > 180, lons_raw - 360, lons_raw)
        tims = ds.groups['data_20'].variables['time'][:] # This is t20
        range_counter = ds.groups['data_20'].groups['ku'].variables['range_ocog'][:]
        altitude= ds.groups['data_20'].variables['altitude'][:]
        wet_tropo= ds.groups['data_20'].variables['model_wet_tropo_cor_measurement_altitude'][:]
        dry_tropo= ds.groups['data_20'].variables['model_dry_tropo_cor_measurement_altitude'][:]
        
        pole_tide_1hz = ds.groups['data_01'].variables['pole_tide'][:]
        timestamps_1hz = ds.groups['data_01'].variables['time'][:] # This is t1
        solid_tide_1hz = ds.groups['data_01'].variables['solid_earth_tide'][:] 
        iono_1hz=ds.groups['data_01'].groups['ku'].variables['iono_cor_gim'][:]
        geoid_1hz = ds.groups['data_01'].variables['geoid'][:]
    except Exception as e:
        ds.close()
        raise RuntimeError(f"SWOT extraction error: File structure does not match user-provided (Jason-3-style) format. Error: {e}")

    # Interpolate 1Hz data to match 20Hz timestamps
    pole_tide_20hz = interpolate_1hz_to_20hz(pole_tide_1hz, timestamps_1hz, tims)
    solid_tide_20hz = interpolate_1hz_to_20hz(solid_tide_1hz, timestamps_1hz, tims)
    iono_20hz=interpolate_1hz_to_20hz(iono_1hz, timestamps_1hz, tims)
    geoid_20hz = interpolate_1hz_to_20hz(geoid_1hz, timestamps_1hz, tims)
    
    ds.close()
            
    df = pd.DataFrame({
        'mission': 'SWOT', # Set mission as SWOT
        'date': file_date,
        'latitude': lats, 
        'longitude': lons, 
        'time_20hz': tims, # Harmonized name
        'altitude': altitude, # Harmonized name
        'range': range_counter, 
        'dry': dry_tropo,
        'wet': wet_tropo,
        'pole':pole_tide_20hz,
        'solid':solid_tide_20hz,
        'iono':iono_20hz,
        'geoid':geoid_20hz
    })
    
    df['date'] = pd.to_datetime(df['date']) # Ensure column is datetime type
    return df

def extract_sentinel6_data(nc_path):
    """Sentinel-6A extractor (same structure and logic as Jason-3)."""
    ds = Dataset(nc_path, "r")
    fname = os.path.basename(nc_path)

    mission = "S6A"

    # --- DATE LOGIC (same philosophy as Jason-3) ---
    file_date = pd.NaT
    try:
    # Sentinel-6 filenames contain YYYYMMDDTHHMMSS
        date_token = next(p for p in fname.split("_") if "T" in p and p[:8].isdigit())
        file_date = datetime.strptime(date_token[:8], "%Y%m%d")
    except Exception:
        print(f"Warning: Sentinel-6 file {fname} - date parsing failed.")

    try:
        g20 = ds.groups["data_20"]
        g01 = ds.groups["data_01"]

        lats = g20.groups["ku"].variables["latitude"][:]
        lons_raw = g20.groups["ku"].variables["longitude"][:]
        lons = np.where(lons_raw > 180, lons_raw - 360, lons_raw)
        tims = g20.groups["ku"].variables["time"][:]

        rng = g20.groups["ku"].variables["range_ocog"][:]
        altitude = g20.groups["ku"].variables["altitude"][:]
        wet = g20.groups["ku"].variables["model_wet_tropo_cor_measurement_altitude"][:]
        dry = g20.groups["ku"].variables["model_dry_tropo_cor_measurement_altitude"][:]
        geoid = g20.groups["ku"].variables["geoid"][:]

        t1 = g01.variables["time"][:]
        pole = interpolate_1hz_to_20hz(g01.variables["pole_tide"][:], t1, tims)
        solid = interpolate_1hz_to_20hz(g01.variables["solid_earth_tide"][:], t1, tims)
        iono = interpolate_1hz_to_20hz(
            g01.groups["ku"].variables["iono_cor_gim"][:], t1, tims
        )

    except Exception as e:
        ds.close()
        raise RuntimeError(f"Sentinel-6 extraction failed: {e}")

    ds.close()

    df = pd.DataFrame({
        "mission": mission,
        "date": file_date,
        "latitude": lats,
        "longitude": lons,
        "time_20hz": tims,
        "altitude": altitude,
        "range": rng,
        "dry": dry,
        "wet": wet,
        "pole": pole,
        "solid": solid,
        "iono": iono,
        "geoid": geoid,
    })

    df["date"] = pd.to_datetime(df["date"])
    return df


def extract_altimetry_data(nc_path):
    f = nc_path.lower()

    # ---------- Sentinel-3 ZIP ----------
    if f.endswith(".sen3.zip"):
        nc_file, tmpdir = extract_s3_zip(nc_path)
        try:
            df = extract_sentinel3_data(nc_file)
        finally:
            shutil.rmtree(tmpdir)
        return df

    # ---------- Sentinel-3 (already extracted) ----------
    if f.endswith("enhanced_measurement.nc"):
        return extract_sentinel3_data(nc_path)

    # ---------- SWOT ----------
    if "swot" in os.path.basename(f):
        return extract_swot_data(nc_path)

    # ---------- Sentinel-6 ----------
    if "sentinel-6" in f or "sentinel6" in f or "s6a" in f or "s6" in os.path.basename(f):
        return extract_sentinel6_data(nc_path)

    # ---------- Jason-3 ----------
    if "ja3" in f or "jason" in f:
        return extract_jason3_or_s6_data(nc_path)

    raise RuntimeError(f"Unknown or unsupported altimetry file: {nc_path}")
