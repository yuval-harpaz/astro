import pandas as pd
import astropy.time
from astroquery.mast import Observations
import numpy as np
import os

def fetch_older():
    # 1. Identify Boundary
    latest_csv_path = 'docs/latest.csv'
    if not os.path.exists(latest_csv_path):
        print(f"Error: {latest_csv_path} not found.")
        return

    prev = pd.read_csv(latest_csv_path)
    # Convert t_obs_release to MJD to find the earliest
    # latest.csv has format like '2026-02-19 15:44:37.995'
    # Use pandas for robust parsing then convert to astropy Time
    release_times_pd = pd.to_datetime(prev['t_obs_release'].dropna())
    if release_times_pd.empty:
        print("Error: No valid release times found in latest.csv")
        return
    
    release_times = astropy.time.Time(release_times_pd.values)
    min_mjd = np.min(release_times.mjd)
    print(f"Earliest release in latest.csv: {astropy.time.Time(min_mjd, format='mjd').iso} (MJD: {min_mjd})")

    # 2. Query MAST
    # We want ALL public images prior to min_mjd. 
    # Since JWST started operations in 2022, we can set a safe start time (e.g. 50000 MJD is way before)
    start_time = 59000.0 # Well before JWST launch/commissioning (approx Mid-2022)
    end_time = min_mjd

    args = {
        'obs_collection': "JWST",
        'calib_level': 3,
        'dataRights': 'public',
        'dataproduct_type': "image"
    }

    print(f"Querying MAST for JWST images between MJD {start_time} and {end_time}...")
    
    # Query images by observation date and by release date
    table_release = Observations.query_criteria(t_obs_release=[start_time, end_time], **args)
    table_min = Observations.query_criteria(t_min=[start_time, end_time], **args)
    
    # Combine results
    table = table_min.to_pandas().merge(table_release.to_pandas(), how='outer')
    print(f"Found {len(table)} raw results.")

    if len(table) == 0:
        print("No older images found.")
        return

    # 3. Filtering logic matching astro_jwst_news_2025.py
    # Instrument filter
    instrument = table['instrument_name'].str.lower().fillna('').values
    inst_ok = [ (('nircam' in x) | ('miri' in x) | ('niriss' in x)) for x in instrument]
    table = table[inst_ok]
    
    # Calibration filter
    # Use fillna to avoid errors with NaN titles
    calibration = np.asarray((table['obs_title'].str.contains("alibration", na=False)) | (table['intentType'] != 'science'))
    science = table[~calibration]
    print(f"After filtering: {len(science)} science images.")

    if len(science) == 0:
        print("No science images found after filtering.")
        return

    # 4. Formatting and sorting
    keep = ['obsid', 'proposal_id', 'proposal_pi', 'target_name', 'instrument_name', 'obs_title', 'jpegURL', 'dataURL', 't_max', 't_obs_release']
    older = science[keep].copy()
    
    # Ensure no duplicates with latest.csv
    prev_ids = set(prev['obsid'].astype(str))
    older = older[~older['obsid'].astype(str).isin(prev_ids)]
    print(f"After removing duplicates with latest.csv: {len(older)} images.")

    # Sort by t_obs_release descending
    older = older.sort_values('t_obs_release', ascending=False, ignore_index=True)

    # Handle jpegURL nulls and mast: prefix
    mast_url = 'https://mast.stsci.edu/portal/Download/file/'
    
    # Matching logic for null jpegs from news script
    nanjpeg = np.where(older['jpegURL'].isnull())[0]
    if len(nanjpeg) > 0:
        for ii in nanjpeg:
            data_url = str(older.at[ii, 'dataURL'])
            if '.fits' in data_url:
                repurl = data_url.replace('.fits', '.jpg')
            else:
                repurl = 'docs/broken.jpg'
            older.at[ii, 'jpegURL'] = repurl

    # Convert MJD to UTC ISO for the final CSV
    for col in ['t_obs_release', 't_max']:
        older[col] = older[col].apply(lambda x: astropy.time.Time(x, format='mjd').utc.iso if pd.notnull(x) else x)

    # 5. Save to older.csv
    output_path = 'docs/older.csv'
    older.to_csv(output_path, index=False)
    print(f"Successfully saved {len(older)} records to {output_path}")

if __name__ == "__main__":
    fetch_older()
