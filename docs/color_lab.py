"""Python core of color_lab.html (https://yuval-harpaz.github.io/astro/color_lab.html).

Runs inside Pyodide (web worker), and is also embedded as-is in the stand-alone script the page generates
for running locally (see run_local at the bottom).

Pipeline, following the color bot (astro_jwst_news_color.py) defaults:
  read SCI only (HTTP range requests to the STScI S3 mirror) -> block-average to ~max_mp megapixels
  -> fill holes (vectorized version of astro_fill_holes.hole_func_fill, func='max', 5x5)
  -> reproject to the first chosen layer (WCS + scipy map_coordinates, bilinear)
  -> level_adjust(factor=2) per layer -> mean per channel (green = (R+B)/2 if no green)
  -> NaN to 0 -> grey_zeros(replace=np.max) -> uint8
Reprojected layers are cached per reference, so stretch settings can be changed without reprojecting.
"""
import gc
import os
import re
import time
import urllib.request
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy import ndimage
from scipy.ndimage import map_coordinates, maximum_filter, label
from numpy.lib.stride_tricks import sliding_window_view as _swv

warnings.simplefilter('ignore', FITSFixedWarning)

VERSION = '1.1'  # bump, commit, then tag colorlab-v<VERSION> (minor: fixes and features, major: images look different)

S3 = 'https://stpubdata.s3.amazonaws.com/jwst/public/'
S3_HST = 'https://stpubdata.s3.amazonaws.com/hst/public/'
RAW = {}     # name -> (data float32 2D, WCS, block factor, (full nx, full ny)), downsampled and hole-filled
REPROJ = {}  # (name, ref, align, (dx, dy)) -> float32 2D on ref grid, with a nudge of (dx, dy) ref pixels built in
MOVING = {}  # name -> True for moving targets (HST MTFLAG = T, JWST TARGTYPE = MOVING): align by pixels, not by sky
BAKED = {}   # (name, ref, align, ref shape) -> nudge (dx, dy) in ref pixels built into the reprojection ("Fill edges")
GRID = {}    # ref -> (shape, WCS, signature, names): reference grid extended to cover mosaic tiles (set_grid)
CRMASK = {}  # name -> fraction of cosmic-ray pixels removed in each pixel of the layer (same grid as RAW data)
WCSNAME = {}  # name -> WCSNAME of the SCI header: HST '...GAIA...' = aligned to Gaia, others can be off by ~1 arcsec
_T0 = [time.time()]


def reset_clock():
    _T0[0] = time.time()


def step(msg):
    print(f'[{time.time() - _T0[0]:6.1f}s] {msg}', flush=True)


# {file name: [filter label, wavelength in nm]}, given by the page (or the local script) for files whose name has no
# filter, e.g. HST jfq8jq010_drc.fits. JWST names carry the filter, F444W -> 444 x 10 nm.
FILTERS = {}


def set_filters(filters):
    if hasattr(filters, 'to_py'):
        filters = filters.to_py()
    FILTERS.update({k: [str(v[0]), float(v[1])] for k, v in (filters or {}).items()})


def filt_num(name):
    """Filter number like astro_utils.filt_num: the last [-_]fNNN in the name (f444w -> 444, f2550w -> 2550)."""
    m = re.findall(r'[-_]f(\d{3,4})', name.lower())
    return int(m[-1]) if m else 0


def wavelength(name):
    """Pivot-ish wavelength in nm for sorting and filt05 colors (JWST filter numbers are in units of 10 nm)."""
    if name in FILTERS:
        return FILTERS[name][1]
    return filt_num(name) * 10.0


def short(name):
    """jw07321-o004_t002_nircam_clear-f444w_i2d.fits -> f444w"""
    if name in FILTERS:
        return FILTERS[name][0]
    m = re.findall(r'[-_](f\d{3,4}[a-z]*\d?)', name.lower())
    return m[-1] if m else name


# ---------- stretch, copied from bot_grabber.py ----------
def to1(x):
    return 0.5 + (2.0 * x - 2.0) / (2 * np.sqrt((2.0 * x - 2.0) * (2.0 * x - 2.0) + 1.0))


def expand_highs(x):
    return np.piecewise(x, [x <= 0.9, x > 0.9],
                        [lambda x: x * 0.8 / 0.9, lambda x: 100.0 / 9.0 * (x - 0.9) ** 2 + 0.8 * x / 0.9])


def image_histogram_equalization(image, number_bins=10000, ignore0=False, sample=None):
    # color_lab addition: sample (bool mask) = build the histogram from these pixels only, apply it to all
    shape = image.shape
    image = image.flatten()
    if ignore0:
        include = image > 0
    else:
        include = np.ones(len(image), bool)
    hist_from = include if sample is None else include & sample.flatten()
    image_histogram, bins = np.histogram(image[hist_from], number_bins, density=True)
    cdf = image_histogram.cumsum()
    cdf = cdf / cdf[-1]
    image[include] = np.interp(image[include], bins[:-1], cdf)
    return image.reshape(shape)


def nanmask(arr):
    mask = np.isnan(arr)
    if np.sum(mask) > 0:
        arr[mask] = 0
    else:
        mask = None
    return arr, mask


def level_adjust(fits_arr, negative='consider', lims=[0.03, 0.98], factor=4.0, ignore0=False, sample=None):
    # color_lab addition: sample (bool mask, e.g. a galaxy nucleus) = take the limits and the histogram
    # equalization from these pixels only, and apply them to the whole image
    fits_arr, mask = nanmask(fits_arr)
    hist_dat = fits_arr.flatten() if sample is None else fits_arr[sample]
    if negative == 'consider':
        hist_dat = hist_dat[np.nonzero(hist_dat)]
    else:
        hist_dat = hist_dat[hist_dat > 0]
    nonzeros = np.abs(np.sign(fits_arr))
    if lims[1] > 1:
        minval, maxval = lims
    else:
        minval = np.quantile(hist_dat, lims[0])
        maxval = np.quantile(hist_dat, lims[1])
    # color_lab change: pixels below the lower limit are left out of the histogram equalization and stay dark
    # (the original's ignore0 behavior), so the lower limit works for every factor. The original also hard-clipped
    # non-default limits to [0, 1] before to1(), capping highlights at expand_highs(0.75) = 0.667 (170/255);
    # here the highs are clipped only at the 99.8% quantile, as in the original's default path.
    rescaled = (fits_arr - minval) / (maxval - minval)
    if sample is None:
        rescaled_no_outliers = np.maximum(rescaled, 0.0)
        rescaled_no_outliers = np.minimum(rescaled_no_outliers, np.quantile(rescaled_no_outliers, 1.0 - 0.002))
    else:  # outside the sampled range saturates (to1 below still uses the unclipped values)
        rescaled_no_outliers = np.clip(rescaled, 0.0, 1.0)
    ignore0 = True
    if factor is None:
        adjusted = rescaled
    else:
        img_eqd = image_histogram_equalization(rescaled_no_outliers, ignore0=ignore0, sample=sample)
        img_eqd = (pow(img_eqd, factor) + pow(img_eqd, factor * 2) + pow(img_eqd, factor * 4)) / 3.0
        adjusted = expand_highs((img_eqd + to1(rescaled)) * 0.5)
    if mask is not None:
        adjusted[mask] = np.nan
    return np.clip(adjusted * nonzeros, 0.0, 1.0)


def grey_zeros(img, bad=[0, 1, 2], thr=0, replace=np.min):
    # copied from astro_utils.py
    for bd in bad:
        okay = [0, 1, 2]
        okay.pop(bd)
        layer = img[..., bd]
        mask = layer <= thr
        layer[mask] = replace(img[..., okay], 2)[mask]
        img[..., bd] = layer
    return img


def log1(arr):
    # copied from astro_utils.py, shown as "log" in the GUI
    arr[arr < 0] = 0
    arr += 1
    arr = np.log10(arr)
    return arr


def asinh(arr):
    # copied from astro_utils.py
    arr = np.arcsinh(arr)
    return arr


def blc_image(img):
    # copied from astro_utils.py (float version): per channel, subtract the non-zero minimum and divide by max
    for ii in range(3):
        layer = img[..., ii]
        layer[layer <= 0] = np.nan
        layer = layer - np.nanmin(layer)
        layer = layer / np.nanmax(layer)
        layer[np.isnan(layer)] = 0
        img[..., ii] = layer
    return img


# matplotlib's 'hsv' colormap segment data (x, value), so colors match auto_plot without importing matplotlib
_HSV = {'red': [(0.0, 1.0), (0.15873, 1.0), (0.174603, 0.96875), (0.333333, 0.03125), (0.349206, 0.0), (0.666667, 0.0),
                (0.68254, 0.03125), (0.84127, 0.96875), (0.857143, 1.0), (1.0, 1.0)],
        'green': [(0.0, 0.0), (0.15873, 0.9375), (0.174603, 1.0), (0.507937, 1.0), (0.666667, 0.0625), (0.68254, 0.0), (1.0, 0.0)],
        'blue': [(0.0, 0.0), (0.333333, 0.0), (0.349206, 0.0625), (0.507937, 1.0), (0.84127, 1.0), (0.857143, 0.9375), (1.0, 0.09375)]}


def hsv_cmap(x):
    """Same as matplotlib.cm.hsv(x)[:, :3] (256-entry lookup table)."""
    lut_x = np.linspace(0, 1, 256)
    lut = np.stack([np.interp(lut_x, *zip(*_HSV[c])) for c in ['red', 'green', 'blue']], 1)
    return lut[np.clip((np.asarray(x) * 256).astype(int), 0, 255)]


def colors_by_filt(filt, subtract_blue=0.5):
    """Colors of astro_utils.assign_colors_by_filt (method='filt05', colormap='hsv'): long wavelength red,
    short wavelength blue/violet."""
    filt = np.asarray(filt, float)
    filtnorm = filt - filt.min() * subtract_blue
    filtnorm = filtnorm / filtnorm.max()
    return [tuple(c) for c in hsv_cmap((1 - filtnorm) * 0.680)]


def shift_layer(img, dx, dy):
    """Move the image content dx pixels right and dy pixels up (array rows increase upward in FITS)."""
    if not dx and not dy:
        return img
    out = np.full_like(img, np.nan)
    ny, nx = img.shape
    if abs(dx) >= nx or abs(dy) >= ny:
        return out
    out[max(dy, 0):ny + min(dy, 0), max(dx, 0):nx + min(dx, 0)] = img[max(-dy, 0):ny + min(-dy, 0), max(-dx, 0):nx + min(-dx, 0)]
    return out


# ---------- hole filling, vectorized astro_fill_holes.hole_func_fill(func='max') ----------
def largest_cluster_mask(labels, min_size=0.01):
    if labels.max() == 0:
        return np.zeros(labels.shape, bool)
    unique_labels, counts = np.unique(labels[labels > 0], return_counts=True)
    if np.max(counts) < (labels.shape[0] * labels.shape[1] * min_size):
        return np.zeros(labels.shape, bool)
    return labels == unique_labels[np.argmax(counts)]


def fill_holes(img, size=5, n_iter=100):
    """Holes = NaN or <= nanmin, inside the image footprint (per row and per column), excluding the largest
    NaN cluster (the outside). Each iteration sets holes to the max of their valid size x size neighbours."""
    fill_below = np.nanmin(img)
    nan_labels, _ = label(np.isnan(img))
    cluster = largest_cluster_mask(nan_labels)
    del nan_labels
    ny, nx = img.shape
    edge = size // 2 + 1
    interior = np.zeros(img.shape, bool)
    interior[edge:ny - edge, edge:nx - edge] = True
    n_holes0 = None
    for it in range(n_iter):
        img[np.isnan(img)] = fill_below
        img[img <= fill_below] = fill_below
        valid = img > fill_below
        inside = (np.cumsum(valid, 1) > 0) & (np.cumsum(valid[:, ::-1], 1)[:, ::-1] > 0)
        inside &= (np.cumsum(valid, 0) > 0) & (np.cumsum(valid[::-1], 0)[::-1] > 0)
        holes = ~valid & inside & ~cluster & interior
        n = int(holes.sum())
        if n_holes0 is None:
            n_holes0 = n
        if n == 0:
            img[~inside] = np.nan  # like hole_func_fill, which breaks before restoring the outside to fill_below
            break
        mx = maximum_filter(np.where(valid, img, -np.inf), size=size)
        fill = mx[holes]
        fill[~np.isfinite(fill)] = fill_below
        img[holes] = fill
    if cluster.any():
        img[cluster] = np.nan
    return img, n_holes0, it


def fill_gaps(img, thin=15, n_iter=100):
    """HST: missing data is NaN (no_data), and 0 or the minimum can be real (photon counting), so only NaN holes
    are filled: inside the footprint (data on both sides along the row and the column) and either not part of the
    outside (largest NaN cluster) or thinner than `thin` pixels (dead rows and bad columns, which can run into the
    outside). Each iteration gives holes next to data the mean of their valid 3 x 3 neighbours, so a dead row gets
    the level of its surroundings instead of their brightest pixel. Returns img, holes, iterations."""
    nan = np.isnan(img)
    if not nan.any():
        return img, 0, 0
    labels, _ = label(nan)
    outside = largest_cluster_mask(labels)
    del labels
    valid = ~nan
    inside = (np.cumsum(valid, 1) > 0) & (np.cumsum(valid[:, ::-1], 1)[:, ::-1] > 0)
    inside &= (np.cumsum(valid, 0) > 0) & (np.cumsum(valid[::-1], 0)[::-1] > 0)
    thin_nan = nan & ~ndimage.binary_opening(nan, np.ones((thin, thin), bool))
    holes = nan & inside & (~outside | thin_nan)
    n0 = int(holes.sum())
    it = 0
    for it in range(1, n_iter + 1):
        v = np.isfinite(img)
        cnt = ndimage.uniform_filter(v.astype(np.float32), 3)
        tot = ndimage.uniform_filter(np.where(v, img, 0).astype(np.float32), 3)
        new = holes & ~v & (cnt > 0.05)
        if not new.any():
            break
        img[new] = tot[new] / cnt[new]
        holes &= ~new
        if not holes.any():
            break
    return img, n0, it


# ---------- reading FITS SCI with HTTP range requests (used by the local script; the browser does it in JS) ----------
def s3url(fname):
    if not fname.startswith('jw'):  # HST pipeline product, e.g. jfq8jq010_drc.fits
        obs = fname.split('_')[0]
        return f'{S3_HST}{obs[:4]}/{obs}/{fname}'
    prop, obs = fname.split('_')[0].split('-')[:2]
    return f'{S3}{prop}/L3/t/{obs}/{fname}'


def _get(url, a, b):
    req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b}'})
    with urllib.request.urlopen(req) as r:
        return r.read()


def _read_header(url, offset):
    text, pos, chunk = '', offset, 2880 * 16
    while True:
        t = _get(url, pos, pos + chunk - 1).decode('ascii', 'replace')
        for i in range(0, len(t) - 79, 80):
            card = t[i:i + 80]
            text += card
            if card[:8].strip() == 'END':
                hdr = fits.Header.fromstring(text)
                n = int(np.prod([hdr[f'NAXIS{a}'] for a in range(1, hdr['NAXIS'] + 1)])) if hdr['NAXIS'] else 0
                start = offset + int(np.ceil(len(text) / 2880)) * 2880
                return text, hdr, start, n * abs(hdr['BITPIX']) // 8
        pos += len(t)
        if len(t) < chunk:
            raise IOError('END card not found')


def is_moving(primary):
    """Moving target (comet, asteroid, planet): HST MTFLAG = T, JWST TARGTYPE = MOVING. primary: dict-like header."""
    mt = primary.get('MTFLAG', '')
    return mt is True or str(mt).strip().upper() in ('T', 'TRUE') or str(primary.get('TARGTYPE', '')).strip().upper() == 'MOVING'


def set_moving(name, flag):
    MOVING[name] = bool(flag)


def block_factor(nx, ny, max_mp):
    return 1 if not max_mp else max(1, int(np.ceil(np.sqrt(nx * ny / (max_mp * 1e6)))))


def _windows(xp, ys, xs, k):
    """(n, k, k) windows of the padded image xp (pad k // 2) centred at pixels (ys, xs) of the unpadded image.
    Gathered by index, not by a sliding-window view of the whole image, which can exceed the 2 GB array limit
    of the browser's 32-bit Python even though it is only a view."""
    o = np.arange(k)
    return xp[ys[:, None, None] + o[None, :, None], xs[:, None, None] + o[None, None, :]]


def _median_at(x, ys, xs, k, xp=None):
    """k x k median at pixels (ys, xs) of x (edge-padded; xp: x already padded by k // 2)."""
    xp = np.pad(x, k // 2, mode='edge') if xp is None else xp
    return np.median(_windows(xp, ys, xs, k).reshape(len(ys), -1), -1)

def cosmic_mask(d, sig=5.0, objlim=3.0, b=32, grow=3):
    """Boolean mask of cosmic-ray hits (dots and short streaks), a simplified L.A.Cosmic (van Dokkum 2001), meant for
    full-resolution data, where a hit is much sharper than a star: a pixel is a hit when it is sig x the local noise
    above the 5x5 median AND much sharper than the fine structure of real sources (med3 - med7 of med3), which keeps
    stars and nebula detail. Fast: candidates come from a 5x5 box mean, the median and the fine-structure test are
    computed at the candidates only. Hits grow (up to `grow` pixels) into neighbours that are also clearly high.
    0 and NaN are no data (HST drizzled products)."""
    d = np.asarray(d, np.float32)
    good = np.isfinite(d) & (d != 0)
    mask = np.zeros(d.shape, bool)
    if good.sum() < 100:
        return mask
    x = np.where(good, d, np.float32(np.median(d[good])))
    ny, nx = x.shape
    r0 = x - ndimage.uniform_filter(x, 5)
    py_, px_ = (-ny) % b, (-nx) % b
    r = np.pad(np.abs(r0), ((0, py_), (0, px_)), mode='edge').reshape((ny + py_) // b, b, (nx + px_) // b, b)
    blk = 1.4826 * np.median(r.transpose(0, 2, 1, 3).reshape(r.shape[0], r.shape[2], -1), -1)
    noise = ndimage.zoom(blk, b, order=1, grid_mode=True, mode='nearest')[:ny, :nx] if blk.size > 1 else np.full(x.shape, blk.item(), np.float32)
    if noise.shape != x.shape:
        noise = np.pad(noise, ((0, ny - noise.shape[0]), (0, nx - noise.shape[1])), mode='edge')
    noise = np.maximum(noise, 1.4826 * np.median(np.abs(r0[good])) + 1e-12)
    ys, xs = np.nonzero(good & (r0 > 0.8 * sig * noise))
    if not len(ys):
        return mask
    res = x[ys, xs] - _median_at(x, ys, xs, 5)
    n = noise[ys, xs]
    ok = res > sig * n
    ys, xs, res, n = ys[ok], xs[ok], res[ok], n[ok]
    if not len(ys):
        return mask
    xp9 = np.pad(x, 4, mode='edge')
    fine = np.empty(len(ys), np.float32)
    for i in range(0, len(ys), 20000):  # batches: 441 values per candidate
        w = _windows(xp9, ys[i:i + 20000], xs[i:i + 20000], 9)
        m3 = np.median(_swv(w, (3, 3), axis=(1, 2)).reshape(len(w), 7, 7, 9), -1)
        fine[i:i + 20000] = m3[:, 3, 3] - np.median(m3.reshape(len(w), -1), -1)
    fine = np.maximum(fine, 0.1 * n)
    keep = res / fine > objlim
    mask[ys[keep], xs[keep]] = True
    # grow along streaks: neighbours clearly above the 5x5 median of the pixels not yet flagged, up to `grow` times
    for _ in range(grow):
        gy, gx = np.nonzero(ndimage.binary_dilation(mask, np.ones((3, 3), bool)) & ~mask & good)
        if not len(gy):
            break
        w = _windows(np.pad(np.where(mask, np.nan, x), 2, mode='edge'), gy, gx, 5).reshape(len(gy), -1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            hit = x[gy, gx] - np.nanmedian(w, -1) > 2 * noise[gy, gx]
        if not hit.any():
            break
        mask[gy[hit], gx[hit]] = True
    return mask


def clean_strip(d, top=0, bottom=0):
    """Remove cosmic-ray hits from full-resolution rows. d: rows including `top` and `bottom` margin rows borrowed
    from the neighbouring strips (so hits at the strip edges are judged with their neighbours). Hits get the median
    of the other pixels in their 5x5 neighbourhood. Returns (cleaned rows, hit mask) without the margins."""
    d = np.array(d, np.float32)
    m = cosmic_mask(d)
    if m.any():
        x = np.where(m | (d == 0), np.nan, d)
        ys, xs = np.nonzero(m)
        w = _windows(np.pad(x, 2, mode='constant', constant_values=np.nan), ys, xs, 5).reshape(len(ys), -1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            d[ys, xs] = np.nanmedian(w, -1)  # NaN if all 25 are hits or no data
    b = d.shape[0] - bottom
    return d[top:b], m[top:b]


def clean_strip_buf(buf, ny, nx, top=0, bottom=0):
    """For the browser: buf = native float32 rows (ny, nx) as bytes / Uint8Array. Returns (cleaned float32 bytes,
    uint8 mask bytes, number of hit pixels), without the margin rows."""
    if hasattr(buf, 'to_py'):
        buf = buf.to_py()
    d = np.frombuffer(buf, np.float32).reshape(int(ny), int(nx))
    c, m = clean_strip(d, int(top), int(bottom))
    return c.tobytes(), m.astype(np.uint8).tobytes(), int(m.sum())


def skip_cosmic_text(name, header_text, detector=''):
    """skip_cosmic from the SCI header text and the primary header's DETECTOR (for the browser): '' = clean it."""
    return skip_cosmic(name, fits.Header.fromstring(header_text), detector) or ''


CR_MARGIN = 8  # rows borrowed from the neighbouring strips when cleaning a strip (the tests look 4 px away)


def _bin_rows(rows, f, ox):
    """nan-mean of f x f blocks of full-resolution rows (the last block row / column may be partial)."""
    if f == 1:
        return rows
    pad = np.full((-(-rows.shape[0] // f) * f, ox * f), np.nan, np.float32)
    pad[:rows.shape[0], :rows.shape[1]] = rows
    pad = pad.reshape(pad.shape[0] // f, f, ox, f)
    valid = ~np.isnan(pad)
    cnt = valid.sum(axis=(1, 3))
    tot = np.where(valid, pad, 0).sum(axis=(1, 3), dtype=np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).astype(np.float32)


def no_data_from_weight(name, detector):
    """HST photon-counting detectors (ACS/SBC, STIS MAMA): a pixel can be 0 counts, so missing data is read from WHT."""
    return not os.path.basename(name).lower().startswith('jw') and str(detector).strip().upper() in MAMA


LOW_WEIGHT = 0.5  # photon counting: pixels below this fraction of the median weight are mostly 0 counts (biased)


def no_data(name, rows, wht=None):
    """Full-resolution HST rows with missing data as NaN: when the weight map is given (photon-counting detectors,
    where 0 counts is real), weight below LOW_WEIGHT x its median (dead rows and the frame edge, where a pixel is
    0 counts far more often than elsewhere); else exactly 0 (drizzle writes 0 where no exposure had data).
    JWST as is."""
    if os.path.basename(name).lower().startswith('jw'):
        return rows
    rows = np.array(rows, np.float32)
    if wht is not None:
        pos = wht[wht > 0]
        rows[~(wht >= LOW_WEIGHT * (np.median(pos) if pos.size else 0)) | (wht <= 0)] = np.nan
    else:
        rows[rows == 0] = np.nan
    return rows


def read_sci(fname, max_mp, cosmic=False):
    """Download only the SCI extension and block-average it (nan-mean). cosmic: remove specks (cosmic-ray hits,
    hot pixels) at full resolution, strip by strip, before the block-average (where a hit is still much sharper than a star), unless
    the header says the image needs none (skip_cosmic). HST pixels without data (dead rows, bad columns: weight 0
    in the drizzled product) are NaN before the block-average, so they leave no dark lines (no_data).
    Returns data, header text, factor, hit mask (fraction of hit pixels in each block, on the same grid as data;
    None when not cleaned)."""
    url = s3url(fname)
    off = 0
    for ext in range(6):
        text, hdr, start, size = _read_header(url, off)
        if ext == 0:
            MOVING[fname] = is_moving(hdr)
            detector = str(hdr.get('DETECTOR', '') or '')
        if ext > 0 and hdr.get('EXTNAME', '').strip() == 'SCI':
            break
        off = start + int(np.ceil(size / 2880)) * 2880
    else:
        raise IOError('no SCI extension in ' + fname)
    nx, ny = hdr['NAXIS1'], hdr['NAXIS2']
    f = block_factor(nx, ny, max_mp)
    ox, oy = -(-nx // f), -(-ny // f)
    out = np.empty((oy, ox), np.float32)
    wht = None
    if no_data_from_weight(fname, detector):  # zeros are real here: the weight map tells where data is missing
        wtext, whdr, wstart, wsize = _read_header(url, start + int(np.ceil(size / 2880)) * 2880)
        if whdr.get('EXTNAME', '').strip() == 'WHT' and (whdr['NAXIS1'], whdr['NAXIS2'], whdr['BITPIX']) == (nx, ny, -32):
            wht = wstart
            step(f'{short(fname)}: {detector.strip()} counts photons, zeros are real: reading the weight map (WHT, {wsize / 2 ** 20:.0f} MB) for missing data')
    why = skip_cosmic(fname, hdr, detector) if cosmic else 'off'
    if cosmic and why != 'JWST' and why:
        step(f'{short(fname)}: no speck removal ({why})')
    clean = not why
    mask = np.zeros((oy, ox), np.float32) if clean else None
    hits, tc = 0, 0.0
    rows = max(f, (32 * 2 ** 20 // (nx * 4)) // f * f)
    step(f'{short(fname)}: downloading SCI {nx}x{ny} ({size / 2 ** 20:.0f} MB), block-average /{f} -> {ox}x{oy}')
    for y0 in range(0, ny, rows):
        y1 = min(ny, y0 + rows)
        a, b = (max(0, y0 - CR_MARGIN), min(ny, y1 + CR_MARGIN)) if clean else (y0, y1)
        strip = np.frombuffer(_get(url, start + a * nx * 4, start + b * nx * 4 - 1), '>f4').reshape(b - a, nx)
        w = None if wht is None else np.frombuffer(_get(url, wht + a * nx * 4, wht + b * nx * 4 - 1), '>f4').reshape(b - a, nx)
        strip = no_data(fname, strip, w)
        if clean:
            t = time.time()
            strip, m = clean_strip(strip, y0 - a, b - y1)
            hits += int(m.sum())
            tc += time.time() - t
            mask[y0 // f:y0 // f + -(-(y1 - y0) // f)] = _bin_rows(m.astype(np.float32), f, ox)
        out[y0 // f:y0 // f + -(-(y1 - y0) // f)] = _bin_rows(strip.astype(np.float32), f, ox)
    if clean:
        step(f'{short(fname)}: removed specks (cosmic rays, hot pixels) at full resolution, {hits} pixels ({hits / (nx * ny) * 100:.2f}%), {tc:.1f}s')
    return out, text, f, mask


# ---------- layers ----------
def binned_wcs(header_text, factor):
    """WCS for an image block-averaged by `factor` (new pixel = mean of factor x factor old pixels)."""
    hdr = fits.Header.fromstring(header_text)
    wcs = WCS(hdr, naxis=2)
    if factor > 1:
        # 1-based: old = (new - 1) * f + (f + 1) / 2
        wcs.wcs.crpix = (wcs.wcs.crpix - (factor + 1) / 2) / factor + 1
        if wcs.wcs.has_cd():
            wcs.wcs.cd = wcs.wcs.cd * factor
        else:
            wcs.wcs.cdelt = wcs.wcs.cdelt * factor
        wcs.wcs.set()
    return wcs


def add_layer(name, buf, ny, nx, header_text, factor, fill=True, mask=None):
    """buf: bytes-like (or Pyodide JsProxy of a Uint8Array) of native float32, row-major (ny, nx).
    mask: same layout, fraction of cosmic-ray pixels per block (None = not cleaned)."""
    if hasattr(buf, 'to_py'):
        buf = buf.to_py()
    data = np.frombuffer(buf, dtype=np.float32).reshape(int(ny), int(nx)).copy()
    if mask is not None:
        if hasattr(mask, 'to_py'):
            mask = mask.to_py()
        mask = np.frombuffer(mask, dtype=np.float32).reshape(int(ny), int(nx)).copy()
    _add(name, data, header_text, int(factor), fill, mask)
    return data.nbytes


MAMA = ('SBC', 'FUV-MAMA', 'NUV-MAMA')  # photon-counting UV detectors (ACS/SBC, STIS): no cosmic-ray hits


def skip_cosmic(name, hdr, detector=''):
    """Why an image gets no speck removal (cosmic rays, hot pixels; None = clean it), from its headers:
    JWST i2d products are already clean; on IR detectors (WFC3/IR, NICMOS) the pixels are large for the star image,
    so star cores are as sharp as specks (and the pipeline already rejects hits); photon-counting MAMA detectors
    (ACS/SBC, STIS; detector: DETECTOR of the primary header) have no hits, and their single-photon noise peaks
    would be flattened, darkening faint light. Optical / UV CCD images are cleaned also when drizzled from several
    exposures: specks remain where only one exposure covers (edges, chip gaps), and at full resolution the
    well-covered stars and clusters are left alone."""
    base = os.path.basename(name).lower()
    if base.startswith('jw'):
        return 'JWST'
    try:
        scale = 3600 * float(np.hypot(hdr.get('CD1_1', 0), hdr.get('CD2_1', 0)))
    except (TypeError, ValueError):
        scale = 0
    detector = str(detector or hdr.get('DETECTOR', '') or '').strip().upper()
    if detector in MAMA:
        return f'{detector}: photon counting, no specks'
    if base.startswith('n') or detector == 'IR' or (base.startswith('i') and scale > 0.1):
        return 'infrared: star cores are as sharp as specks'
    return None


def _add(name, data, header_text, factor, fill, mask=None):
    """mask: fraction of cosmic-ray pixels per pixel of data (read_sci), kept as layer 'cosmic' (see reproject_to)."""
    hdr = fits.Header.fromstring(header_text)
    WCSNAME[name] = str(hdr.get('WCSNAME', '') or '').strip()
    if mask is not None:
        CRMASK[name] = mask
    else:
        CRMASK.pop(name, None)
    if fill:
        t = time.time()
        data, n, it = fill_holes(data) if os.path.basename(name).lower().startswith('jw') else fill_gaps(data)
        step(f'{short(name)}: fill holes, {n} hole pixels, {it} iterations, {time.time() - t:.1f}s')
    RAW[name] = (data, binned_wcs(header_text, factor), factor, (int(hdr['NAXIS1']), int(hdr['NAXIS2'])))


def has_layer(name):
    return name in RAW


def grid(ref):
    """(shape, WCS) of the reference grid: the reference layer's own, or extended to cover mosaic tiles."""
    return GRID[ref][:2] if ref in GRID else (RAW[ref][0].shape, RAW[ref][1])


def gridscale(ref):
    """Pixel size (arcsec) of the reference grid."""
    return float(np.mean(proj_plane_pixel_scales(grid(ref)[1])) * 3600)


def set_grid(ref, names, align='sky', mosaic=True, max_factor=4):
    """Mosaics of tiles that do not overlap (e.g. a galaxy in two halves from different visits): with mosaic on and
    sky alignment, the reference grid is extended to cover all images of filters that have two or more images, so a
    tile outside the reference image is not cut away. Same orientation and pixel size as the reference; when the
    extended grid would exceed max_factor x the reference's pixels, it is coarsened by an integer factor.
    A single image of another filter never extends the grid (the reference still crops it)."""
    old = GRID.get(ref)
    new = None
    if align == 'sky' and mosaic and ref in RAW:
        groups = {}
        for n in names:
            groups.setdefault((short(n).lower(), round(wavelength(n))), []).append(n)
        tiles = [n for g in groups.values() if len(g) > 1 for n in g if n in RAW]
        if tiles:
            (ny, nx), rw = RAW[ref][0].shape, RAW[ref][1]
            xs, ys = [-0.5, nx - 0.5], [-0.5, ny - 0.5]
            for n in tiles:
                data, w = RAW[n][:2]
                fin = np.isfinite(data)
                rows, cols = np.nonzero(fin.any(1))[0], np.nonzero(fin.any(0))[0]
                if not len(rows):
                    continue
                cx = np.array([cols[0], cols[-1], cols[0], cols[-1]], float) + np.array([-0.5, 0.5, -0.5, 0.5])
                cy = np.array([rows[0], rows[0], rows[-1], rows[-1]], float) + np.array([-0.5, -0.5, 0.5, 0.5])
                x, y = rw.world_to_pixel_values(*w.pixel_to_world_values(cx, cy))
                xs += list(x)
                ys += list(y)
            x0, x1 = int(np.floor(min(xs) + 0.5)), int(np.ceil(max(xs) - 0.5))
            y0, y1 = int(np.floor(min(ys) + 0.5)), int(np.ceil(max(ys) - 0.5))
            if x0 < -1 or y0 < -1 or x1 > nx or y1 > ny:  # tiles reach beyond the reference image
                x0, y0, x1, y1 = min(x0, 0), min(y0, 0), max(x1, nx - 1), max(y1, ny - 1)
                W, H = x1 - x0 + 1, y1 - y0 + 1
                k = max(1, int(np.ceil(np.sqrt(W * H / (max_factor * nx * ny)))))
                w = rw.deepcopy()
                # 0-based: old p = x0 + q * k + (k - 1) / 2 for new pixel q
                w.wcs.crpix = [(w.wcs.crpix[0] - 1 - x0 - (k - 1) / 2) / k + 1, (w.wcs.crpix[1] - 1 - y0 - (k - 1) / 2) / k + 1]
                if k > 1:
                    if w.wcs.has_cd():
                        w.wcs.cd = w.wcs.cd * k
                    else:
                        w.wcs.cdelt = w.wcs.cdelt * k
                w.wcs.set()
                shape = (-(-H // k), -(-W // k))
                new = (shape, w, (shape, tuple(np.round(w.wcs.crpix, 6)), k), list(names))
    sig = lambda g: g[2] if g else None
    if sig(old) != sig(new):
        for k_ in [k_ for k_ in REPROJ if k_[1] == ref]:
            del REPROJ[k_]
        for k_ in [k_ for k_ in BAKED if k_[1] == ref]:
            del BAKED[k_]
    if new is None:
        GRID.pop(ref, None)
    else:
        GRID[ref] = new
        if sig(old) != sig(new):
            (gy, gx), k = new[0], new[2][2]
            step(f'mosaic: reference grid extended to cover all tiles, {gx}x{gy} ({gx * gy / 1e6:.1f} MP)' +
                 (f', {k}x coarser than {short(ref)} to stay within {max_factor}x its size' if k > 1 else ''))
    return new


def reproject_to(name, ref, align='sky', shift=(0, 0), chunk=256, what='data', cache=True):
    """Layer `name` on the grid of `ref`, moved by shift = (dx, dy) ref pixels (right, up), so a nudge brings in the
    layer's own data instead of empty edges. align 'sky': by WCS; 'pixels': centers matched and the pixel scale of ref,
    ignoring the sky position (moving targets, where the stars move between images).
    what 'cosmic': the layer's cosmic-ray mask (fraction of removed pixels, CRMASK) instead of its data, processed
    the same way (not cached; None when the layer was not cleaned). cache False: not kept (one-off grids)."""
    shift = (int(shift[0]), int(shift[1]))
    key = (name, ref, align, shift, GRID[ref][2] if ref in GRID else None)
    cosmic = what == 'cosmic'
    if cosmic and name not in CRMASK:
        return None
    if not cosmic and key in REPROJ:
        step(f'{short(name)}: reprojected layer cached')
        return REPROJ[key]
    data, wcs = (CRMASK[name] if cosmic else RAW[name][0]), RAW[name][1]
    own = name == ref and ref not in GRID  # the reference on its own grid
    if own and shift == (0, 0):
        out = data
    else:
        t = time.time()
        (ny, nx), ref_wcs = grid(ref)
        ly, lx = data.shape
        s = pixscale(ref) / pixscale(name)  # layer pixels per ref pixel
        out = np.empty((ny, nx), np.float32)
        xx = np.arange(nx, dtype=np.float64) - shift[0]
        for y0 in range(0, ny, chunk):
            y1 = min(ny, y0 + chunk)
            gx, gy = np.meshgrid(xx, np.arange(y0, y1, dtype=np.float64) - shift[1])
            if align == 'pixels' or own:
                px, py = (gx - (nx - 1) / 2) * s + (lx - 1) / 2, (gy - (ny - 1) / 2) * s + (ly - 1) / 2
            else:
                lon, lat = ref_wcs.pixel_to_world_values(gx, gy)
                px, py = wcs.world_to_pixel_values(lon, lat)
            out[y0:y1] = map_coordinates(data, [py, px], order=1, cval=np.nan, prefilter=False)
        how = 'pixel grid' if align == 'pixels' else 'grid'
        step(f'{short(name)}: {"cosmic-ray mask " if cosmic else ""}reprojected onto {short(ref)} {how} ({nx}x{ny})' + (f', moved {shift[0]:+d}, {shift[1]:+d} px' if any(shift) else '') + f', {time.time() - t:.1f}s')
    if cosmic or not cache:
        return out
    for k in [k for k in REPROJ if k[:3] == key[:3] and k != key]:  # keep one version per layer (shift, grid)
        del REPROJ[k]
    REPROJ[key] = out
    return out


def gaia_aligned(name):
    """HST: WCSNAME says so (e.g. IDC_...-FIT_REL_GAIAeDR3); JWST pipeline products are tied to Gaia."""
    return os.path.basename(name).startswith('jw') or 'GAIA' in WCSNAME.get(name, '').upper()


def _point_sources(img, edge=4):
    """High-pass image in noise units, clipped to 0..50: stars and compact knots stand out, smooth light and sky
    are ~0. Pixels near the footprint edge are 0 so the edge itself does not pull the alignment to no shift."""
    good = np.isfinite(img)  # not "!= 0": photon-counting UV images (ACS/SBC) have real zeros
    if good.sum() < 100:
        return None
    x = np.where(good, img, np.median(img[good])).astype(np.float32)
    r = x - ndimage.median_filter(x, 5)
    s = 1.4826 * np.median(np.abs(r[good])) + 1e-12
    r = np.clip(r / s, 0, 50)
    r[~ndimage.binary_erosion(good, iterations=edge)] = 0
    return r


def _xcorr(a, b, maxs):
    """(dx, dy, snr, at_limit): moving b by (dx, dy) pixels (right, up) best matches a; snr = peak height over the scatter
    of the correlation within +-maxs."""
    ny, nx = a.shape
    shape = (ny + maxs, nx + maxs)  # zero padding: no wrap-around
    c = np.fft.irfft2(np.fft.rfft2(a, shape) * np.conj(np.fft.rfft2(b, shape)), shape)
    w = np.roll(np.roll(c, maxs, 0), maxs, 1)[:2 * maxs + 1, :2 * maxs + 1]
    iy, ix = np.unravel_index(np.argmax(w), w.shape)
    med = np.median(w)
    snr = (w[iy, ix] - med) / (1.4826 * np.median(np.abs(w - med)) + 1e-12)
    edge = ix in (0, 2 * maxs) or iy in (0, 2 * maxs)
    fx = fy = 0.0
    if not edge:  # sub-pixel peak: parabola through the peak and its neighbours
        l, m, r = w[iy, ix - 1], w[iy, ix], w[iy, ix + 1]
        fx = float(np.clip(0.5 * (l - r) / (l - 2 * m + r - 1e-12), -0.5, 0.5))
        l, r = w[iy - 1, ix], w[iy + 1, ix]
        fy = float(np.clip(0.5 * (l - r) / (l - 2 * m + r - 1e-12), -0.5, 0.5))
    return ix - maxs + fx, iy - maxs + fy, float(snr), edge


def auto_align(names, ref=None, align='sky', max_arcsec=3.0, min_snr=6.0):  # false matches score 3-4.5
    """Nudges (arcsec, right / up, as compose_rgb's shifts) that line up the stars and compact knots of all layers,
    found by cross-correlation. Layers whose WCS is aligned to Gaia stay put (they agree with each other) and each
    other layer is matched to the one of them that it resembles most (clearest match; a far-UV image may match the
    infrared better than the optical). Without Gaia-aligned layers, the reference stays put.
    Each layer is compared on the finer of its own grid and the reference grid: a small, sharp image (e.g. ACS/SBC)
    keeps its knots, which a coarse reference grid would blur away; the shift is then turned into the reference frame.
    Returns {name: [dx, dy]} for all names (anchors [0, 0]); a layer whose match is not clear keeps [0, 0] and is
    reported in the log."""
    names = list(names.to_py() if hasattr(names, 'to_py') else names)
    names = [n for n in dict.fromkeys(names) if n in RAW]
    if not names:
        return {}
    ref = ref if ref in names else names[0]
    if align not in ('sky', 'pixels'):
        align = 'pixels' if any(MOVING.get(n) for n in names) else 'sky'
    t = time.time()
    anchors = [n for n in names if gaia_aligned(n)] or [ref]
    scale = gridscale(ref)
    step(f'auto-align: staying put: {", ".join(short(n) for n in anchors)}' +
         (' (WCS aligned to Gaia)' if gaia_aligned(anchors[0]) else ' (reference; no layer is aligned to Gaia)') +
         f', search +-{max_arcsec:.1f}"')
    out = {n: [0.0, 0.0] for n in names}
    on_ref = {}  # anchor -> point sources on the reference grid
    for n in names:
        if n in anchors:
            continue
        own = align == 'sky' and pixscale(n) < 0.8 * scale  # compare on the layer's own, finer grid
        grid = n if own else ref
        gs = pixscale(grid)
        maxs = max(3, int(np.ceil(max_arcsec / gs)))
        b = _point_sources(RAW[n][0] if own else reproject_to(n, ref, align))
        best = None
        for anc in anchors:
            if own:
                a = _point_sources(reproject_to(anc, n, align, cache=False))
            else:
                if anc not in on_ref:
                    on_ref[anc] = _point_sources(reproject_to(anc, ref, align))
                a = on_ref[anc]
            if a is None or b is None:
                continue
            r = _xcorr(a, b, maxs)
            if best is None or (not r[3] and (best[1][3] or r[2] > best[1][2])):
                best = (anc, r)
        wcs = WCSNAME.get(n, '') or 'no WCSNAME'
        where = f'on its own grid ({gs:.3f}"/px)' if own else 'on the reference grid'
        if best is None:
            step(f'auto-align: {short(n)}: no overlap with {", ".join(short(a) for a in anchors)}, left as is')
            continue
        anc, (dx, dy, snr, edge) = best
        if snr < min_snr or edge:
            step(f'auto-align: {short(n)}: no clear match {where} (best confidence {snr:.0f}, with {short(anc)}'
                 f'{", at the search limit" if edge else ""}), left as is [{wcs}]')
            continue
        if own:  # the shift on the layer's grid -> on the sky -> on the reference grid
            w, rw = RAW[n][1], grid(ref)[1]
            ny, nx = RAW[n][0].shape
            cx, cy = (nx - 1) / 2, (ny - 1) / 2
            lon, lat = w.pixel_to_world_values(np.array([cx, cx + dx]), np.array([cy, cy + dy]))
            rx, ry = rw.world_to_pixel_values(lon, lat)
            sx, sy = float(rx[1] - rx[0]) * scale, float(ry[1] - ry[0]) * scale
        else:
            sx, sy = dx * scale, dy * scale
        out[n] = [sx, sy]
        step(f'auto-align: {short(n)}: {sx:+.2f}", {sy:+.2f}" ({sx / scale:+.1f}, {sy / scale:+.1f} px of the reference), '
             f'confidence {snr:.0f} with {short(anc)} {where} [{wcs}]')
    step(f'auto-align: {time.time() - t:.1f}s')
    return out

def clear():
    RAW.clear()
    GRID.clear()
    CRMASK.clear()
    REPROJ.clear()
    BAKED.clear()
    gc.collect()


def forget(keep, max_raw=8):
    """Free memory before a compose. `keep` = names about to be used (keep[0] is the reference).
    Raw layers are kept LRU up to max_raw so toggling a checkbox does not re-download;
    reprojected layers are kept only for the current reference."""
    keep = list(keep)
    for name in keep:  # mark as recently used
        if name in RAW:
            RAW[name] = RAW.pop(name)
    extra = [k for k in RAW if k not in keep]
    for k in extra[:max(0, len(RAW) - max(max_raw, len(keep)))]:
        del RAW[k]
        CRMASK.pop(k, None)
    ref = keep[0] if keep else None
    for k in [k for k in REPROJ if k[1] != ref or k[0] not in RAW]:
        del REPROJ[k]
    for r in [r for r in GRID if r != ref or r not in RAW]:
        del GRID[r]
    gc.collect()


DEFAULTS = dict(method='rgb', func='none', factor=2.0, lims=[0.03, 0.98], grey=True, shifts=None, ref=None, area=None,
                subtract_blue=0.5, align='auto', fill_edges=False, mosaic=True)


def area_mask(area, shape):
    """area = [x0, x1, y0, y1] as fractions of the reference image (FITS orientation, y up) -> bool mask."""
    ny, nx = shape
    x0, x1 = sorted(float(v) for v in area[:2])
    y0, y1 = sorted(float(v) for v in area[2:])
    ix0, ix1 = int(np.floor(x0 * nx)), max(int(np.ceil(x1 * nx)), int(np.floor(x0 * nx)) + 1)
    iy0, iy1 = int(np.floor(y0 * ny)), max(int(np.ceil(y1 * ny)), int(np.floor(y0 * ny)) + 1)
    m = np.zeros(shape, bool)
    m[max(iy0, 0):min(iy1, ny), max(ix0, 0):min(ix1, nx)] = True
    return m


def stretch(lay, name, func='none', factor=2.0, lims=(0.03, 0.98), sample=None):
    t = time.time()
    lims = [float(lims[0]), float(lims[1])]
    if not np.any(np.isfinite(lay) & (lay != 0)):
        step(f'{short(name)}: no data on the reference grid (it does not overlap the reference image)')
        return np.full(lay.shape, np.nan, np.float32)
    if func == 'log':
        lay = log1(lay)
    elif func == 'asinh':
        lay = asinh(lay)
    factor = None if factor in (None, 0, 'linear') else float(factor)
    if sample is not None:
        v = lay[sample]
        if np.count_nonzero(v[np.isfinite(v)]) < 10:
            step(f'{short(name)}: selected area is empty in this layer, using the whole image')
            sample = None
    out = level_adjust(lay, lims=lims, factor=factor, sample=sample)
    desc = (f'{func} -> ' if func in ('log', 'asinh') else '') + f'level_adjust(factor={"None (linear)" if factor is None else f"{factor:g}"}, lims={lims[0] * 100:g}%-{lims[1] * 100:g}%{" of selected area" if sample is not None else ""})'
    step(f'{short(name)}: {desc}, {time.time() - t:.1f}s')
    return out


def compose_rgb(red, green, blue, **opts):
    """Returns uint8 (ny, nx, 3) with FITS orientation (origin lower-left). Reference = first layer listed.
    opts: method 'rgb' (layers per channel, like the bot) | 'filt05' (all layers colored by wavelength, like auto_plot)
    | 'natural' (each layer in the color of its wavelength, opts colors {name: (r, g, b)} 0-1),
    func 'none' | 'log' | 'asinh', factor 1-4 | None (linear), lims [low, high] quantiles, grey (grey_zeros),
    shifts {name: [dx, dy]} in arcsec (right, up), so nudges keep their size on the sky at any resolution or reference,
    ref: name of the layer whose grid all others are reprojected to (like auto_plot's reproject_to); None = first listed,
    area: [x0, x1, y0, y1] fractions of the reference image (y up): limits and histogram are sampled there only."""
    o = dict(DEFAULTS)
    o.update({k: v for k, v in opts.items() if v is not None or k == 'factor'})
    set_filters(o.get('filters'))
    red, green, blue = list(red), list(green), list(blue)
    allnames = list(dict.fromkeys(red + green + blue))
    if not allnames:
        raise ValueError('no layers chosen')
    ref = o['ref'] if o['ref'] in allnames else allnames[0]
    align = o['align'] if o['align'] in ('sky', 'pixels') else ('pixels' if any(MOVING.get(n) for n in allnames) else 'sky')
    set_grid(ref, allnames, align, o['mosaic'])
    ny, nx = grid(ref)[0]
    shifts = o['shifts'] or {}
    sample = area_mask(o['area'], (ny, nx)) if o['area'] else None
    if sample is not None:
        step(f'limits sampled from a {int(sample.any(1).sum())}x{int(sample.any(0).sum())} px area')
    scale = 1.0 / gridscale(ref)  # arcsec -> pixels of the reference grid
    hst = [n for n in allnames if WCSNAME.get(n) and not os.path.basename(n).startswith('jw')]
    loose = [n for n in hst if not gaia_aligned(n)]
    if loose and len(hst) > 1 and align == 'sky':
        step('note: not aligned to Gaia, may be off by up to ~1": ' + ', '.join(f'{short(n)} ({WCSNAME[n]})' for n in loose) +
             '. Auto-align can line them up.')
    if o['align'] not in ('sky', 'pixels'):
        step(f'align: {align}' + (' (moving target: the stars move between images, so layers are matched by pixels)' if align == 'pixels' else ''))
    # nudges in reference pixels; the reference keeps the frame, so a nudge of the reference moves the others the opposite way
    px = {n: (int(round(float(shifts.get(n, (0, 0))[0]) * scale)), int(round(float(shifts.get(n, (0, 0))[1]) * scale))) for n in allnames}
    eff = {n: (px[n][0] - px[ref][0], px[n][1] - px[ref][1]) for n in allnames}
    if any(px[ref]):
        step(f'{short(ref)} is the reference: its nudge moves the other layers {-px[ref][0]:+d}, {-px[ref][1]:+d} px instead')

    def placed(name):  # reprojected onto the reference grid and nudged, not stretched
        dx, dy = eff[name]
        bkey = (name, ref, align, (ny, nx))
        if o['fill_edges']:  # build the current nudge into the reprojection: the edges get the layer's own data
            BAKED[bkey] = (dx, dy)
        bx, by = BAKED.get(bkey, (0, 0))
        lay = reproject_to(name, ref, align, (bx, by)).copy()
        rx, ry = dx - bx, dy - by
        if rx or ry:  # fast: move the reprojected layer (uncovered edges stay empty until Fill edges)
            lay = shift_layer(lay, rx, ry)
            step(f'{short(name)}: nudged {rx:+d} px right, {ry:+d} px up')
        return lay

    def layer(name):
        return stretch(placed(name), name, o['func'], o['factor'], o['lims'], sample)

    def units(names):
        """Layers to stretch: with mosaic, images of the same filter are one layer (list of names), else one each."""
        if not o['mosaic']:
            return [[n] for n in names]
        groups = {}
        for n in names:
            groups.setdefault((short(n).lower(), round(wavelength(n))), []).append(n)
        return list(groups.values())

    def unit_layer(group):
        """One stretched layer per filter. Several images of a filter are a mosaic: on the reference grid, each pixel
        is the median of the images that have data there (0 and NaN are no data), so segments complete
        each other, and the stretch (level_adjust) sees the whole mosaic at once, not each segment on its own."""
        if len(group) == 1:
            return layer(group[0])
        t = time.time()
        stack = np.empty((len(group), ny, nx), np.float32)
        for i, n in enumerate(group):
            stack[i] = placed(n)
        stack[stack == 0] = np.nan  # 0 = no data (negative values are sky noise and stay)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            lay = np.nanmedian(stack, 0).astype(np.float32)
        cover = [float(np.isfinite(stack[i]).mean() * 100) for i in range(len(group))]
        del stack
        step(f'{short(group[0])}: mosaic of {len(group)} images (median where they overlap; each covers '
             f'{", ".join(f"{c:.0f}%" for c in cover)}, together {np.isfinite(lay).mean() * 100:.0f}%), {time.time() - t:.1f}s')
        return stretch(lay, group[0], o['func'], o['factor'], o['lims'], sample)

    if o['method'] == 'natural':  # each filter in its own color (the page's spectrum bar), added, hues kept
        cols = o.get('colors') or {}
        rgb = np.zeros((ny, nx, 3), np.float64)
        for group in units(sorted(allnames, key=wavelength)):
            name = group[0]
            col = cols.get(name, (1.0, 1.0, 1.0))
            lay = unit_layer(group)
            lay[np.isnan(lay)] = 0
            for ich in range(3):
                rgb[..., ich] += lay * col[ich]
            step(f'{short(name)}: natural color RGB({col[0]:.2f}, {col[1]:.2f}, {col[2]:.2f})')
        top = np.quantile(rgb.max(-1), 0.999)
        rgb = np.clip(rgb / (top if top > 0 else 1), 0, 1)
        step('natural: added, then scaled together (99.9% of the brightest channel = white), so each filter keeps its hue')
    elif o['method'] == 'filt05':
        groups = units(sorted(allnames, key=wavelength))
        cols = colors_by_filt([wavelength(g[0]) for g in groups], float(o['subtract_blue']))
        step(f'filt05: Blue % = {float(o["subtract_blue"]) * 100:g}')
        rgb = np.zeros((ny, nx, 3), np.float64)
        for group, col in zip(groups, cols):
            name = group[0]
            lay = unit_layer(group)
            lay[np.isnan(lay)] = 0
            for ich in range(3):
                rgb[..., ich] += lay * col[ich]
            step(f'{short(name)}: color by wavelength (filt05) = RGB({col[0]:.2f}, {col[1]:.2f}, {col[2]:.2f})')
        rgb /= np.max(rgb)
        rgb = blc_image(rgb)
        step('filt05: normalize, then stretch each color channel between its lowest and highest value')
    else:
        rgb = np.zeros((ny, nx, 3), np.float32)
        for ich, names in enumerate([red, green, blue]):
            if not names:
                continue
            acc = np.zeros((ny, nx), np.float32)
            cnt = np.zeros((ny, nx), np.uint16)
            groups = units(names)
            for group in groups:  # nanmean: a pixel covered by any of the layers keeps data (a plain mean needs all of them)
                lay = unit_layer(group)
                ok = ~np.isnan(lay)
                acc[ok] += lay[ok]
                cnt += ok
                del lay, ok
            with np.errstate(invalid='ignore', divide='ignore'):
                rgb[..., ich] = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
            del acc, cnt
            step(f'{"RGB"[ich]} = ' + (f'nanmean of {len(groups)} layers' if len(groups) > 1 else short(names[0])))
        if not green and red and blue:  # like the bot with 2 files
            rgb[..., 1] = (rgb[..., 0] + rgb[..., 2]) / 2
            step('G = (R + B) / 2')
    missing = np.isnan(rgb)  # outside a channel's footprint (rgb method); filt05 already set those to 0
    rgb[missing] = 0
    if o['grey'] and o['method'] in ('filt05', 'natural'):
        rgb = grey_zeros(rgb, replace=np.max)
        step('grey missing: fill zero pixels of each channel with the max of the other two')
    elif o['grey']:
        # color_lab change: like grey_zeros(replace=np.max), but only where a channel has no data. The bot's grey_zeros
        # also fills pixels that are 0 because of noise (e.g. sky-subtracted MIRI after log, which sets negatives to 0),
        # painting faint areas with the other channels' values (purple / magenta speckle with a NIRCam blue).
        for ich in range(3):
            others = np.max(rgb[..., [k for k in range(3) if k != ich]], -1)
            rgb[..., ich][missing[..., ich]] = others[missing[..., ich]]
        step(f'grey missing: fill pixels outside a channel\'s footprint with max of the other two '
             f'({", ".join(f"{c} {missing[..., i].mean() * 100:.0f}%" for i, c in enumerate("RGB"))})')
    out = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    del rgb
    gc.collect()
    step(f'RGB image {nx}x{ny} ready')
    return out


def pixscale(name):
    """Pixel size (arcsec) of a loaded, downsampled layer."""
    return float(np.mean(proj_plane_pixel_scales(RAW[name][1])) * 3600)


def ref_info(name):
    """(block factor, CRVAL1, CRVAL2, pixel scale arcsec) of a loaded layer as the reference grid."""
    _, wcs, f, _ = RAW[name]
    return f, float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1]), gridscale(name)


def footprint(name):
    """[RA, Dec, radius] in degrees: a circle around the center of the reference grid of a loaded layer that covers
    its corners (for SIMBAD)."""
    (ny, nx), w = grid(name)
    ra, dec = (float(v) for v in w.pixel_to_world_values((nx - 1) / 2, (ny - 1) / 2))
    cra, cdec = w.pixel_to_world_values(np.array([-0.5, nx - 0.5, -0.5, nx - 0.5]), np.array([-0.5, -0.5, ny - 0.5, ny - 0.5]))
    r0, d0, r1, d1 = np.radians(ra), np.radians(dec), np.radians(cra), np.radians(cdec)
    sep = np.degrees(np.arccos(np.clip(np.sin(d0) * np.sin(d1) + np.cos(d0) * np.cos(d1) * np.cos(r1 - r0), -1, 1)))
    return [ra, dec, float(np.max(sep))]


def sky_to_pix(name, ra, dec):
    """Pixel x, y (0-based, FITS orientation) on the reference grid of a loaded layer; NaN outside the image or its
    data (with an extended mosaic grid: outside the data of all its images)."""
    if hasattr(ra, 'to_py'):
        ra, dec = ra.to_py(), dec.to_py()
    ra, dec = np.atleast_1d(np.asarray(ra, float)), np.atleast_1d(np.asarray(dec, float))
    (ny, nx), w = grid(name)
    x, y = w.world_to_pixel_values(ra, dec)
    x, y = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
    ix, iy = np.round(x), np.round(y)
    inside = np.isfinite(x) & np.isfinite(y) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ok = np.zeros(len(x), bool)
    for n in (GRID[name][3] if name in GRID else [name]):  # data of any image there
        if n not in RAW:
            continue
        data, lw = RAW[n][:2]
        lx, ly = lw.world_to_pixel_values(ra, dec)
        jx, jy = np.round(np.atleast_1d(lx)), np.round(np.atleast_1d(ly))
        on = inside & np.isfinite(jx) & np.isfinite(jy) & (jx >= 0) & (jx < data.shape[1]) & (jy >= 0) & (jy < data.shape[0])
        ok[on] |= np.isfinite(data[jy[on].astype(int), jx[on].astype(int)])
    x[~ok], y[~ok] = np.nan, np.nan
    return [x.tolist(), y.tolist()]


def layer_info():
    """{name: [full nx, full ny, block factor, full-resolution pixel scale arcsec]} for all loaded layers."""
    return {k: [v[3][0], v[3][1], v[2], pixscale(k) / v[2]] for k, v in RAW.items()}


def size_report(names, ref, max_mp=0):
    """Warn when the reference grid is much larger than the smallest layer (e.g. reprojecting MIRI to NIRCam SW)."""
    ny, nx = RAW[ref][0].shape
    mp = nx * ny / 1e6
    smallest = min(RAW[n][0].size for n in names) / 1e6
    step(f'reference grid {short(ref)}: {nx}x{ny} = {mp:.1f} MP' + ('' if max_mp else f', about {mp * 0.1:.1f} GB RAM to process'))
    if mp > 1.5 * smallest:
        step(f'note: {mp / smallest:.1f}x the pixels of the smallest layer; a lower-resolution reference saves RAM')


def compose(red, green, blue, opts=None):
    """For the browser: returns (rgba bytes flipped for canvas, height, width). opts: dict, see compose_rgb."""
    opts = opts.to_py() if hasattr(opts, 'to_py') else dict(opts or {})
    rgb = compose_rgb(red, green, blue, **opts)
    ny, nx = rgb.shape[:2]
    rgba = np.empty((ny, nx, 4), np.uint8)
    rgba[..., :3] = rgb[::-1]  # FITS origin is lower-left, canvas is upper-left
    rgba[..., 3] = 255
    return rgba.tobytes(), ny, nx


def brightness_contrast(rgb, brightness=0, contrast=0):
    """Same as the page's Brightness / Contrast sliders (-100..100) on a uint8 image: contrast scales around
    mid-grey by 3 ** (contrast / 100) (1/3 to 3), brightness adds up to +-128 levels."""
    k = 3.0 ** (contrast / 100.0)
    lut = np.clip(np.floor((np.arange(256) - 127.5) * k + 127.5 + brightness * 1.28 + 0.5), 0, 255).astype(np.uint8)
    return lut[rgb]


def run_local(red, green, blue, max_mp=0, fill=True, out='color.jpg', credit='', cosmic=True, **opts):
    """Stand-alone: download, process and save a jpg. max_mp=0 means full resolution. opts: see compose_rgb."""
    reset_clock()
    set_filters(opts.get('filters'))
    for name in dict.fromkeys(list(red) + list(green) + list(blue)):
        data, text, f, mask = read_sci(name, max_mp, cosmic)
        _add(name, data, text, f, fill, mask)
    names = list(dict.fromkeys(list(red) + list(green) + list(blue)))
    size_report(names, opts.get('ref') if opts.get('ref') in names else names[0], max_mp)
    brightness, contrast = float(opts.pop('brightness', 0) or 0), float(opts.pop('contrast', 0) or 0)
    rgb = compose_rgb(red, green, blue, **{**opts, 'fill_edges': True})  # nudges built into the reprojection
    if brightness or contrast:
        rgb = brightness_contrast(rgb, brightness, contrast)
        step(f'brightness {brightness:+g}, contrast {contrast:+g}')
    from PIL import Image
    Image.fromarray(rgb[::-1]).save(out, quality=95)
    step(f'saved {out}')
    if credit:
        ref = opts.get('ref') if opts.get('ref') in names else names[0]
        _, c1, c2, _ = ref_info(ref)
        if 'CRVAL' not in credit:
            credit = credit.replace('.\nCredits', f'. CRVAL: {c1:.6f}, {c2:.6f}\nCredits', 1)
        print('\n' + credit)
