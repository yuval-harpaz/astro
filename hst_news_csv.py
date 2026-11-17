"""HST level-3 images for color_lab.html and hst_latest_release.html.

Metadata comes from the CADC TAP service (fast; MAST was very slow in Nov 2026 and neither allows browser
queries). Only products that exist on the public STScI S3 mirror (which the browser can read) are kept:
pipeline drizzled products, <obs_id>_drc.fits (CTE corrected) or else <obs_id>_drz.fits, with a preview jpg
(<file>.jpg; for WFPC2 <obs_id>_drw.jpg).
Products whose SCI extension is empty (NAXIS = 0, seen in some old drizzled files) are skipped.

Files (docs/hst/):
    YYYY.csv      all images released in that year
    latest.csv    images released in the last LATEST_DAYS days (what the pages load first)
    targets.csv   target_name, years, ra, dec, rad  (index for searching the year files by name or position;
                  targets spread over more than 1 degree get one row per 0.5 degree sky cell)

Coordinates (degrees): s_ra, s_dec = center of the footprint polygon from CADC, s_rad = center to farthest vertex,
as in coords_csv.py for the JWST CSVs.

Usage:
    python hst_news_csv.py --days 10        # daily update (GitHub Action): last 10 days -> year file(s), latest, targets
    python hst_news_csv.py --year 2025      # back-fill one year (queried month by month)
"""
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from astropy.time import Time

from coords_csv import sci_header

TAP = 'https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus/sync'
S3 = 'https://stpubdata.s3.amazonaws.com/hst/public/'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs', 'hst')
LATEST_DAYS = 30
COLUMNS = ['obsid', 'proposal_id', 'proposal_pi', 'target_name', 'instrument_name', 'obs_title', 'jpegURL',
           't_max', 't_obs_release', 'dataURL', 'filters', 's_ra', 's_dec', 's_rad']
INSTRUMENTS = ('ACS/WFC', 'ACS/HRC', 'ACS/SBC', 'WFC3/UVIS', 'WFC3/IR', 'WFPC2', 'WFPC2/WFC', 'WFPC2/PC')  # no NICMOS: its files are not on the S3 mirror


def query(start, end, retries=3):
    """Science images (calibration level 3) released in [start, end) from CADC, as a DataFrame."""
    adql = f"""SELECT O.observationID, O.target_name, O.proposal_id, O.proposal_pi, O.proposal_title, O.instrument_name,
        P.energy_bandpassName, P.dataRelease, P.time_bounds_upper, P.position_bounds
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


def _unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)


def sky_center(ra, dec):
    """Mean direction of points (degrees) -> ra, dec, and the largest distance from it to a point."""
    v = _unit(np.asarray(ra, float), np.asarray(dec, float))
    c = v.mean(0)
    c /= np.linalg.norm(c)
    cra, cdec = np.degrees(np.arctan2(c[1], c[0])) % 360, np.degrees(np.arcsin(c[2]))
    return cra, cdec, float(np.degrees(np.arccos(np.clip(v @ c, -1, 1))).max())


def polygon_center(bounds):
    """CADC position_bounds ('polygon ra1 dec1 ra2 dec2 ...', possibly several) -> (ra, dec, radius) or blanks."""
    nums = [float(x) for x in str(bounds).replace('polygon', ' ').replace('circle', ' ').split() if _isnum(x)]
    if len(nums) < 6:
        return '', '', ''
    ra, dec, rad = sky_center(nums[0::2], nums[1::2])
    return f'{ra:.6f}', f'{dec:.6f}', f'{rad:.5f}'


def _isnum(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


_session = requests.Session()
_session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=32, pool_maxsize=32))


def has_image(url):
    """True if the SCI extension holds an image (some old drizzled products have an empty SCI, NAXIS = 0)."""
    hdr = sci_header(_session, url)
    return hdr is not None and hdr.get('NAXIS', 0) >= 2 and hdr.get('NAXIS1', 0) > 1 and hdr.get('NAXIS2', 0) > 1


def s3_file(obs_id, instrument):
    """Name of the drizzled product on S3 with an image in SCI (drc preferred for ACS/WFC and WFC3/UVIS), or None."""
    folder = f'{S3}{obs_id[:4]}/{obs_id}/'
    for kind in (['drc', 'drz'] if instrument.startswith(('ACS/WFC', 'WFC3/UVIS')) else ['drz', 'drc']):
        name = f'{obs_id}_{kind}.fits'
        for attempt in range(3):
            try:
                if requests.head(folder + name, timeout=30).status_code == 200 and has_image(folder + name):
                    return name
                break
            except requests.RequestException:
                time.sleep(2)
    return None


def preview_url(obs_id, file, instrument):
    """Preview jpg on S3: <file>.jpg, but WFPC2 folders only have <obs_id>_drw.jpg (and _drw_thumb.jpg)."""
    name = f'{obs_id}_drw.jpg' if instrument.startswith('WFPC2') else file.replace('.fits', '.jpg')
    return f'{S3}{obs_id[:4]}/{obs_id}/{name}'


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
        'jpegURL': [preview_url(o, f, i) for o, f, i in zip(df['observationID'], df['file'], df['instrument_name'])],
        't_max': t_max,
        't_obs_release': df['dataRelease'].str.replace('T', ' ').values,
        'dataURL': ['mast:HST/product/' + f for f in df['file']],
        'filters': df['energy_bandpassName'].str.split(';').str[0].values,
    })
    out[['s_ra', 's_dec', 's_rad']] = [polygon_center(b) for b in df['position_bounds']]
    return out.sort_values('t_obs_release', ascending=False, ignore_index=True)


def merge_into(path, new):
    old = pd.read_csv(path, dtype=str, keep_default_na=False) if os.path.exists(path) else pd.DataFrame(columns=COLUMNS)
    both = pd.concat([new.fillna("").astype(str), old], ignore_index=True).drop_duplicates('obsid', keep='first')
    both = both.sort_values('t_obs_release', ascending=False, ignore_index=True)
    both.to_csv(path, index=False)
    return both


def write_latest():
    now = datetime.now(timezone.utc)
    years = sorted({now.year, (now - timedelta(days=LATEST_DAYS)).year})
    frames = [pd.read_csv(os.path.join(OUT, f'{y}.csv'), dtype=str) for y in years if os.path.exists(os.path.join(OUT, f'{y}.csv'))]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    df = df[[c for c in COLUMNS if c in df.columns]]
    cutoff = (now - timedelta(days=LATEST_DAYS)).strftime('%Y-%m-%d')
    df = df[df['t_obs_release'] >= cutoff].sort_values('t_obs_release', ascending=False)
    df.to_csv(os.path.join(OUT, 'latest.csv'), index=False)
    print(f'latest.csv: {len(df)} rows since {cutoff}')


def write_targets(split_deg=1.0, cell_deg=0.5):
    """target_name, years, and a circle (ra, dec, rad) covering its images, for name and position searches.
    Targets spread over more than split_deg (planets, comets, generic parallel-field names such as ANY or PAR)
    get one row per cell_deg sky cell, so a position search does not match them everywhere. The pages merge
    the years of rows with the same name."""
    pts = {}
    for f in sorted(os.listdir(OUT)):
        if f[:4].isdigit() and f.endswith('.csv'):
            df = pd.read_csv(os.path.join(OUT, f), dtype=str, keep_default_na=False)
            for t, g in df.groupby('target_name'):
                lst = pts.setdefault(t, [])
                has = g['s_ra'] != '' if 's_ra' in g else pd.Series(False, index=g.index)
                lst.extend((f[:4], float(a), float(d), float(r)) for a, d, r in zip(g.loc[has, 's_ra'], g.loc[has, 's_dec'], g.loc[has, 's_rad']))
                if (~has).any():
                    lst.append((f[:4], None, None, None))

    def circle(p):
        cra, cdec, _ = sky_center(p[:, 0], p[:, 1])
        sep = np.degrees(np.arccos(np.clip(_unit(p[:, 0], p[:, 1]) @ _unit(cra, cdec), -1, 1)))
        return cra, cdec, float((np.atleast_1d(sep) + p[:, 2]).max())

    rows = []
    for t, lst in pts.items():
        years = sorted({y for y, *_ in lst})
        pos = [(y, a, d, r) for y, a, d, r in lst if a is not None]
        if not pos:
            rows.append((t, ';'.join(years), '', '', ''))
            continue
        p = np.array([(a, d, r) for _, a, d, r in pos])
        cra, cdec, rad = circle(p)
        if rad <= split_deg:
            rows.append((t, ';'.join(years), f'{cra:.5f}', f'{cdec:.5f}', f'{rad:.4f}'))
            continue
        nopos = sorted({y for y, a, *_ in lst if a is None})
        cells = {}
        for (y, a, d, r) in pos:
            key = (int(np.floor(d / cell_deg)), int(np.floor(a * max(np.cos(np.radians(d)), 0.01) / cell_deg)))
            cells.setdefault(key, []).append((y, a, d, r))
        for i, c in enumerate(cells.values()):
            q = np.array([(a, d, r) for _, a, d, r in c])
            cra, cdec, rad = circle(q)
            ys = sorted({y for y, *_ in c} | (set(nopos) if i == 0 else set()))
            rows.append((t, ';'.join(ys), f'{cra:.5f}', f'{cdec:.5f}', f'{rad:.4f}'))
    pd.DataFrame(rows, columns=['target_name', 'years', 'ra', 'dec', 'rad']).to_csv(os.path.join(OUT, 'targets.csv'), index=False)
    print(f'targets.csv: {len(pts)} targets, {len(rows)} rows')


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
