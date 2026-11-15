"""HST level-3 images for color_lab.html and hst_latest_release.html.

Metadata comes from the CADC TAP service (fast; MAST was very slow in Nov 2026 and neither allows browser
queries). Only products that exist on the public STScI S3 mirror (which the browser can read) are kept:
pipeline drizzled products, <obs_id>_drc.fits (CTE corrected) or else <obs_id>_drz.fits, with a preview jpg.

Files (docs/hst/):
    YYYY.csv      all images released in that year
    latest.csv    images released in the last LATEST_DAYS days (what the pages load first)
    targets.csv   target_name, years  (index for searching the year files)

Usage:
    python hst_news_csv.py --days 10        # daily update (GitHub Action): last 10 days -> year file(s), latest, targets
    python hst_news_csv.py --year 2025      # back-fill one year (queried month by month)
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from astropy.time import Time

TAP = 'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync'
S3 = 'https://stpubdata.s3.amazonaws.com/hst/public/'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', 'hst')
LATEST_DAYS = 30
COLUMNS = ['obsid', 'proposal_id', 'proposal_pi', 'target_name', 'instrument_name', 'obs_title', 'jpegURL',
           't_max', 't_obs_release', 'dataURL', 'filters']
INSTRUMENTS = ('ACS/WFC', 'ACS/HRC', 'ACS/SBC', 'WFC3/UVIS', 'WFC3/IR', 'WFPC2', 'WFPC2/WFC', 'WFPC2/PC', 'NICMOS')


def query(start, end, retries=3):
    """Science images (calibration level 3) released in [start, end) from CADC, as a DataFrame."""
    adql = f"""SELECT O.observationID, O.target_name, O.proposal_id, O.proposal_pi, O.proposal_title, O.instrument_name,
        P.energy_bandpassName, P.dataRelease, P.time_bounds_upper
        FROM caom2.Observation O JOIN caom2.Plane P ON O.obsID = P.obsID
        WHERE O.collection = 'HST' AND O.intent = 'science' AND P.calibrationLevel = 3 AND P.dataProductType = 'image'
        AND P.dataRelease >= '{start:%Y-%m-%dT%H:%M:%S}' AND P.dataRelease < '{end:%Y-%m-%dT%H:%M:%S}'"""
    for attempt in range(retries):
        try:
            r = requests.get(TAP, params={'LANG': 'ADQL', 'FORMAT': 'csv', 'QUERY': adql}, timeout=900)
            r.raise_for_status()
            from io import StringIO
            return pd.read_csv(StringIO(r.text), dtype=str)
        except Exception as e:
            print(f'  CADC query {start:%Y-%m-%d}..{end:%Y-%m-%d} failed ({e}), attempt {attempt + 1}')
            time.sleep(10)
    raise RuntimeError('CADC query failed')


def s3_file(obs_id, instrument):
    """Name of the drizzled product on S3 (drc preferred), or None."""
    folder = f'{S3}{obs_id[:4]}/{obs_id}/'
    for kind in (['drc', 'drz'] if instrument.startswith(('ACS/WFC', 'WFC3/UVIS')) else ['drz', 'drc']):
        name = f'{obs_id}_{kind}.fits'
        for attempt in range(3):
            try:
                if requests.head(folder + name, timeout=30).status_code == 200:
                    return name
                break
            except requests.RequestException:
                time.sleep(2)
    return None


def to_rows(df):
    """CADC rows -> page rows (same columns as the JWST CSVs plus filters), only files that exist on S3."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=COLUMNS)
    df = df[~df['observationID'].str.startswith('hst_')]  # HAP products are not on the S3 mirror
    df = df[df['instrument_name'].isin(INSTRUMENTS)]
    df = df[df['energy_bandpassName'].fillna('').str.match(r'^F\d{3}')]  # no grisms, ramps, detection images
    df = df.drop_duplicates('observationID')
    if len(df) == 0:
        return pd.DataFrame(columns=COLUMNS)
    with ThreadPoolExecutor(32) as ex:
        files = list(ex.map(s3_file, df['observationID'], df['instrument_name']))
    df = df.assign(file=files)
    print(f'  {len(df)} candidates, {df["file"].notna().sum()} on S3')
    df = df[df['file'].notna()]
    t_max = Time(pd.to_numeric(df['time_bounds_upper']).values, format='mjd').iso
    out = pd.DataFrame({
        'obsid': df['observationID'].values,
        'proposal_id': df['proposal_id'].values,
        'proposal_pi': df['proposal_pi'].values,
        'target_name': df['target_name'].values,
        'instrument_name': df['instrument_name'].values,
        'obs_title': df['proposal_title'].values,
        'jpegURL': [f'{S3}{o[:4]}/{o}/{f.replace(".fits", ".jpg")}' for o, f in zip(df['observationID'], df['file'])],
        't_max': t_max,
        't_obs_release': df['dataRelease'].str.replace('T', ' ').values,
        'dataURL': ['mast:HST/product/' + f for f in df['file']],
        'filters': df['energy_bandpassName'].str.split(';').str[0].values,
    })
    return out.sort_values('t_obs_release', ascending=False, ignore_index=True)


def merge_into(path, new):
    old = pd.read_csv(path, dtype=str) if os.path.exists(path) else pd.DataFrame(columns=COLUMNS)
    both = pd.concat([new.astype(str), old], ignore_index=True).drop_duplicates('obsid', keep='first')
    both = both.sort_values('t_obs_release', ascending=False, ignore_index=True)
    both.to_csv(path, index=False)
    return both


def write_latest():
    now = datetime.now(timezone.utc)
    years = sorted({now.year, (now - timedelta(days=LATEST_DAYS)).year})
    frames = [pd.read_csv(os.path.join(OUT, f'{y}.csv'), dtype=str) for y in years if os.path.exists(os.path.join(OUT, f'{y}.csv'))]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    cutoff = (now - timedelta(days=LATEST_DAYS)).strftime('%Y-%m-%d')
    df = df[df['t_obs_release'] >= cutoff].sort_values('t_obs_release', ascending=False)
    df.to_csv(os.path.join(OUT, 'latest.csv'), index=False)
    print(f'latest.csv: {len(df)} rows since {cutoff}')


def write_targets():
    years = {}
    for f in sorted(os.listdir(OUT)):
        if f[:4].isdigit() and f.endswith('.csv'):
            for t in pd.read_csv(os.path.join(OUT, f), dtype=str, usecols=['target_name'])['target_name'].dropna().unique():
                years.setdefault(t, []).append(f[:4])
    pd.DataFrame({'target_name': list(years), 'years': [';'.join(v) for v in years.values()]}).to_csv(
        os.path.join(OUT, 'targets.csv'), index=False)
    print(f'targets.csv: {len(years)} targets')


def update(start, end, step_days):
    """Query [start, end) in steps, merge into the year files."""
    t = start
    while t < end:
        t1 = min(end, t + timedelta(days=step_days))
        t0 = time.time()
        rows = to_rows(query(t, t1))
        print(f'{t:%Y-%m-%d}..{t1:%Y-%m-%d}: {len(rows)} images ({time.time() - t0:.0f} s)')
        for year, part in rows.groupby(rows['t_obs_release'].str[:4]):
            merge_into(os.path.join(OUT, f'{year}.csv'), part)
        t = t1


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, help='update with images released in the last N days')
    ap.add_argument('--year', type=int, help='back-fill one release year, month by month')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    if a.year:
        update(datetime(a.year, 1, 1), min(now, datetime(a.year + 1, 1, 1)), 31)
    else:
        update(now - timedelta(days=a.days or 10), now + timedelta(hours=1), 31)
    write_latest()
    write_targets()
