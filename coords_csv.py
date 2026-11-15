"""Add sky coordinates to the image CSVs (docs/latest.csv, docs/older.csv, docs/hst/*.csv).

Columns (degrees): s_ra, s_dec = center of the image array, s_rad = half its diagonal (center to farthest corner).
They come from the SCI header of each FITS file on the public STScI S3 mirror (a few kB per file with HTTP Range
requests), so MAST is not needed. Only rows with empty s_ra are processed, so running it after every news update
is cheap. Files not on S3 yet stay empty and are tried again next time.

Usage:
    python coords_csv.py docs/latest.csv docs/older.csv
"""
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

warnings.simplefilter('ignore', FITSFixedWarning)
S3 = 'https://stpubdata.s3.amazonaws.com/'
BLOCK = 2880
COLS = ['s_ra', 's_dec', 's_rad']


def s3url(fname):
    """Same paths as docs/color_lab_worker.js: JWST jw01234-o001_..._i2d.fits, HST jfq8jq010_drc.fits."""
    if fname.startswith('jw'):
        prop, obs = fname.split('_')[0].split('-')[:2]
        return f'{S3}jwst/public/{prop}/L3/t/{obs}/{fname}'
    obs = fname.split('_')[0]
    return f'{S3}hst/public/{obs[:4]}/{obs}/{fname}'


def _range(session, url, a, b):
    for attempt in range(3):
        try:
            r = session.get(url, headers={'Range': f'bytes={a}-{b}'}, timeout=60)
            if r.status_code in (403, 404):
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    return None


def sci_header(session, url):
    """Header of the first extension named SCI (JWST i2d: ext 1; HST drz/drc: ext 1)."""
    off, chunk = 0, BLOCK * 16
    for ext in range(4):
        text, pos = '', off
        while True:
            data = _range(session, url, pos, pos + chunk - 1)
            if data is None:
                return None
            text += data.decode('ascii', 'replace')
            pos += len(data)
            iend = next((i for i in range(0, len(text) - 79, 80) if text[i:i + 8].strip() == 'END'), None)
            if iend is not None:
                break
            if len(data) < chunk:
                return None
        hdr = fits.Header.fromstring(text[:iend + 80])
        if ext > 0 and str(hdr.get('EXTNAME', '')).strip() == 'SCI':
            return hdr
        nbytes = abs(hdr.get('BITPIX', 8)) // 8
        for k in range(1, hdr.get('NAXIS', 0) + 1):
            nbytes *= hdr[f'NAXIS{k}']
        if hdr.get('NAXIS', 0) == 0:
            nbytes = 0
        off += int(np.ceil((iend + 80) / BLOCK)) * BLOCK + int(np.ceil(nbytes / BLOCK)) * BLOCK
    return None


def footprint(hdr):
    """(ra, dec, radius) in degrees: array center and the distance to its farthest corner."""
    w = WCS(hdr).celestial
    nx, ny = hdr['NAXIS1'], hdr['NAXIS2']
    ra, dec = (float(v) for v in w.pixel_to_world_values((nx - 1) / 2, (ny - 1) / 2))
    cra, cdec = w.pixel_to_world_values(np.array([-0.5, nx - 0.5, -0.5, nx - 0.5]), np.array([-0.5, -0.5, ny - 0.5, ny - 0.5]))
    r0, d0, r1, d1 = np.radians(ra), np.radians(dec), np.radians(cra), np.radians(cdec)
    sep = np.degrees(np.arccos(np.clip(np.sin(d0) * np.sin(d1) + np.cos(d0) * np.cos(d1) * np.cos(r1 - r0), -1, 1)))
    return ra % 360, dec, float(np.max(sep))


def file_of(row):
    url = row['dataURL'] if isinstance(row.get('dataURL'), str) and row['dataURL'].endswith('.fits') else \
        str(row['jpegURL']).rsplit('.', 1)[0] + '.fits'
    return url.split('/')[-1]


def fill(path, workers=32):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in COLS:
        if c not in df.columns:
            df[c] = ''
    todo = np.where(df['s_ra'].str.strip() == '')[0]
    print(f'{path}: {len(df)} rows, {len(todo)} without coordinates')
    if len(todo) == 0:
        return
    files = [file_of(df.iloc[i]) for i in todo]
    session = requests.Session()
    session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers))

    def one(fname):
        try:
            hdr = sci_header(session, s3url(fname))
            return footprint(hdr) if hdr is not None else None
        except Exception as e:
            print(f'  {fname}: {e}')
            return None

    t0, done = time.time(), 0
    with ThreadPoolExecutor(workers) as ex:
        for i, res in zip(todo, ex.map(one, files)):
            done += 1
            if res is not None:
                df.at[i, 's_ra'], df.at[i, 's_dec'], df.at[i, 's_rad'] = f'{res[0]:.6f}', f'{res[1]:.6f}', f'{res[2]:.5f}'
            if done % 1000 == 0:
                print(f'  {done}/{len(todo)} ({time.time() - t0:.0f} s)')
    ok = (df['s_ra'].str.strip() != '').sum()
    print(f'  {path}: {ok}/{len(df)} rows with coordinates ({time.time() - t0:.0f} s)')
    df.to_csv(path, index=False)


if __name__ == '__main__':
    for p in sys.argv[1:] or ['docs/latest.csv', 'docs/older.csv']:
        if os.path.exists(p):
            fill(p)
