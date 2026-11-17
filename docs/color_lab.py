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

warnings.simplefilter('ignore', FITSFixedWarning)

S3 = 'https://stpubdata.s3.amazonaws.com/jwst/public/'
S3_HST = 'https://stpubdata.s3.amazonaws.com/hst/public/'
RAW = {}     # name -> (data float32 2D, WCS, block factor, (full nx, full ny)), downsampled and hole-filled
REPROJ = {}  # (name, ref, align, (dx, dy)) -> float32 2D on ref grid, with a nudge of (dx, dy) ref pixels built in
MOVING = {}  # name -> True for moving targets (HST MTFLAG = T, JWST TARGTYPE = MOVING): align by pixels, not by sky
BAKED = {}   # (name, ref, align, ref shape) -> nudge (dx, dy) in ref pixels built into the reprojection ("Fill edges")
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


def clean_cosmic(d, sig=5.0, objlim=3.0, iters=3):
    """Remove cosmic-ray hits (dots and short streaks) from one image, a simplified L.A.Cosmic (van Dokkum 2001):
    a pixel is a hit when it is sig x the local noise above the 5x5 median AND much sharper than the fine structure
    of real sources (med3 - med7 of med3), which keeps stars and nebula detail. Hits (grown by one pixel) get the
    5x5 median. HST images made from one or two exposures are full of hits; JWST i2d products are already clean.
    Returns the cleaned image and the number of replaced pixels."""
    from numpy.lib.stride_tricks import sliding_window_view as swv
    d = d.astype(np.float32, copy=True)
    good = np.isfinite(d)
    if not good.any():
        return d, 0
    fillv = np.float32(np.nanmedian(d))
    total = np.zeros(d.shape, bool)
    ny, nx = d.shape
    b = 32
    for _ in range(iters):
        x = np.where(good, d, fillv)
        med5 = ndimage.median_filter(x, 5)
        resid = x - med5
        # local noise: robust scatter of the residual in 32 px blocks, interpolated
        py_, px_ = (-ny) % b, (-nx) % b
        r = np.pad(np.abs(resid), ((0, py_), (0, px_)), mode='edge').reshape((ny + py_) // b, b, (nx + px_) // b, b)
        blk = 1.4826 * np.median(r.transpose(0, 2, 1, 3).reshape(r.shape[0], r.shape[2], -1), -1)
        noise = ndimage.zoom(blk, b, order=1)[:ny, :nx]
        noise = np.maximum(noise, 1.4826 * np.median(np.abs(resid[good])) + 1e-12)
        ys, xs = np.nonzero(good & (resid > sig * noise))
        if not len(ys):
            break
        # fine structure at the candidates only, from a 9x9 window: med3 at the center minus the median of the 7x7 med3
        w = swv(np.pad(x, 4, mode='edge'), (9, 9))[ys, xs]
        m3 = np.median(swv(w, (3, 3), axis=(1, 2)).reshape(len(ys), 7, 7, 9), -1)
        fine = np.maximum(m3[:, 3, 3] - np.median(m3.reshape(len(ys), -1), -1), 0.1 * noise[ys, xs])
        keep = resid[ys, xs] / fine > objlim
        if not keep.any():
            break
        cr = np.zeros(d.shape, bool)
        cr[ys[keep], xs[keep]] = True
        cr = ndimage.binary_dilation(cr) & good & (resid > 2 * noise)
        d[cr] = med5[cr]
        total |= cr
    return d, int(total.sum())


def read_sci(fname, max_mp):
    """Download only the SCI extension and block-average it (nan-mean). Returns data, header text, factor."""
    url = s3url(fname)
    off = 0
    for ext in range(6):
        text, hdr, start, size = _read_header(url, off)
        if ext == 0:
            MOVING[fname] = is_moving(hdr)
        if ext > 0 and hdr.get('EXTNAME', '').strip() == 'SCI':
            break
        off = start + int(np.ceil(size / 2880)) * 2880
    else:
        raise IOError('no SCI extension in ' + fname)
    nx, ny = hdr['NAXIS1'], hdr['NAXIS2']
    f = block_factor(nx, ny, max_mp)
    ox, oy = -(-nx // f), -(-ny // f)
    out = np.empty((oy, ox), np.float32)
    rows = max(f, (32 * 2 ** 20 // (nx * 4)) // f * f)
    step(f'{short(fname)}: downloading SCI {nx}x{ny} ({size / 2 ** 20:.0f} MB), block-average /{f} -> {ox}x{oy}')
    for y0 in range(0, ny, rows):
        y1 = min(ny, y0 + rows)
        strip = np.frombuffer(_get(url, start + y0 * nx * 4, start + y1 * nx * 4 - 1), '>f4').reshape(y1 - y0, nx)
        if f == 1:
            out[y0:y1] = strip
            continue
        pad = np.full((-(-(y1 - y0) // f) * f, ox * f), np.nan, np.float32)
        pad[:y1 - y0, :nx] = strip
        pad = pad.reshape(pad.shape[0] // f, f, ox, f)
        valid = ~np.isnan(pad)
        cnt = valid.sum(axis=(1, 3))
        s = np.where(valid, pad, 0).sum(axis=(1, 3), dtype=np.float64)
        with np.errstate(invalid='ignore', divide='ignore'):
            out[y0 // f:y0 // f + cnt.shape[0]] = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return out, text, f


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


def add_layer(name, buf, ny, nx, header_text, factor, fill=True, cosmic=False):
    """buf: bytes-like (or Pyodide JsProxy of a Uint8Array) of native float32, row-major (ny, nx)."""
    if hasattr(buf, 'to_py'):
        buf = buf.to_py()
    data = np.frombuffer(buf, dtype=np.float32).reshape(int(ny), int(nx)).copy()
    _add(name, data, header_text, int(factor), fill, cosmic)
    return data.nbytes


def _add(name, data, header_text, factor, fill, cosmic=False):
    if cosmic and not os.path.basename(name).startswith('jw'):  # HST only: JWST i2d products are already clean
        t = time.time()
        data, n = clean_cosmic(data)
        step(f'{short(name)}: remove cosmic rays, {n} pixels ({n / data.size * 100:.2f}%), {time.time() - t:.1f}s')
    if fill:
        t = time.time()
        data, n, it = fill_holes(data)
        step(f'{short(name)}: fill holes, {n} hole pixels, {it} iterations, {time.time() - t:.1f}s')
    hdr = fits.Header.fromstring(header_text)
    RAW[name] = (data, binned_wcs(header_text, factor), factor, (int(hdr['NAXIS1']), int(hdr['NAXIS2'])))


def has_layer(name):
    return name in RAW


def reproject_to(name, ref, align='sky', shift=(0, 0), chunk=256):
    """Layer `name` on the grid of `ref`, moved by shift = (dx, dy) ref pixels (right, up), so a nudge brings in the
    layer's own data instead of empty edges. align 'sky': by WCS; 'pixels': centers matched and the pixel scale of ref,
    ignoring the sky position (moving targets, where the stars move between images)."""
    shift = (int(shift[0]), int(shift[1]))
    key = (name, ref, align, shift)
    if key in REPROJ:
        step(f'{short(name)}: reprojected layer cached')
        return REPROJ[key]
    data, wcs = RAW[name][:2]
    if name == ref and shift == (0, 0):
        out = data
    else:
        t = time.time()
        ref_data, ref_wcs = RAW[ref][:2]
        ny, nx = ref_data.shape
        ly, lx = data.shape
        s = pixscale(ref) / pixscale(name)  # layer pixels per ref pixel
        out = np.empty((ny, nx), np.float32)
        xx = np.arange(nx, dtype=np.float64) - shift[0]
        for y0 in range(0, ny, chunk):
            y1 = min(ny, y0 + chunk)
            gx, gy = np.meshgrid(xx, np.arange(y0, y1, dtype=np.float64) - shift[1])
            if align == 'pixels' or name == ref:
                px, py = (gx - (nx - 1) / 2) * s + (lx - 1) / 2, (gy - (ny - 1) / 2) * s + (ly - 1) / 2
            else:
                lon, lat = ref_wcs.pixel_to_world_values(gx, gy)
                px, py = wcs.world_to_pixel_values(lon, lat)
            out[y0:y1] = map_coordinates(data, [py, px], order=1, cval=np.nan, prefilter=False)
        how = 'pixel grid' if align == 'pixels' else 'grid'
        step(f'{short(name)}: reprojected onto {short(ref)} {how} ({nx}x{ny})' + (f', moved {shift[0]:+d}, {shift[1]:+d} px' if any(shift) else '') + f', {time.time() - t:.1f}s')
    for k in [k for k in REPROJ if k[:3] == key[:3] and k != key]:  # keep one version per layer
        del REPROJ[k]
    REPROJ[key] = out
    return out


def clear():
    RAW.clear()
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
    ref = keep[0] if keep else None
    for k in [k for k in REPROJ if k[1] != ref or k[0] not in RAW]:
        del REPROJ[k]
    gc.collect()


DEFAULTS = dict(method='rgb', func='none', factor=2.0, lims=[0.03, 0.98], grey=True, shifts=None, ref=None, area=None,
                subtract_blue=0.5, align='auto', fill_edges=False)


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
    ny, nx = RAW[ref][0].shape
    shifts = o['shifts'] or {}
    sample = area_mask(o['area'], (ny, nx)) if o['area'] else None
    if sample is not None:
        step(f'limits sampled from a {int(sample.any(1).sum())}x{int(sample.any(0).sum())} px area')
    scale = 1.0 / pixscale(ref)  # arcsec -> pixels of the reference grid
    align = o['align'] if o['align'] in ('sky', 'pixels') else ('pixels' if any(MOVING.get(n) for n in allnames) else 'sky')
    if o['align'] not in ('sky', 'pixels'):
        step(f'align: {align}' + (' (moving target: the stars move between images, so layers are matched by pixels)' if align == 'pixels' else ''))
    # nudges in reference pixels; the reference keeps the frame, so a nudge of the reference moves the others the opposite way
    px = {n: (int(round(float(shifts.get(n, (0, 0))[0]) * scale)), int(round(float(shifts.get(n, (0, 0))[1]) * scale))) for n in allnames}
    eff = {n: (px[n][0] - px[ref][0], px[n][1] - px[ref][1]) for n in allnames}
    if any(px[ref]):
        step(f'{short(ref)} is the reference: its nudge moves the other layers {-px[ref][0]:+d}, {-px[ref][1]:+d} px instead')

    def layer(name):
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
        return stretch(lay, name, o['func'], o['factor'], o['lims'], sample)

    if o['method'] == 'natural':  # each filter in its own color (the page's spectrum bar), added, hues kept
        cols = o.get('colors') or {}
        rgb = np.zeros((ny, nx, 3), np.float64)
        for name in sorted(allnames, key=wavelength):
            col = cols.get(name, (1.0, 1.0, 1.0))
            lay = layer(name)
            lay[np.isnan(lay)] = 0
            for ich in range(3):
                rgb[..., ich] += lay * col[ich]
            step(f'{short(name)}: natural color RGB({col[0]:.2f}, {col[1]:.2f}, {col[2]:.2f})')
        top = np.quantile(rgb.max(-1), 0.999)
        rgb = np.clip(rgb / (top if top > 0 else 1), 0, 1)
        step('natural: added, then scaled together (99.9% of the brightest channel = white), so each filter keeps its hue')
    elif o['method'] == 'filt05':
        names = sorted(allnames, key=wavelength)
        cols = colors_by_filt([wavelength(n) for n in names], float(o['subtract_blue']))
        step(f'filt05: Blue % = {float(o["subtract_blue"]) * 100:g}')
        rgb = np.zeros((ny, nx, 3), np.float64)
        for name, col in zip(names, cols):
            lay = layer(name)
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
            for name in names:
                acc += layer(name)
            rgb[..., ich] = acc / len(names)
            step(f'{"RGB"[ich]} = ' + (f'mean of {len(names)} layers' if len(names) > 1 else short(names[0])))
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
    """(block factor, CRVAL1, CRVAL2, pixel scale arcsec) of a loaded layer."""
    _, wcs, f, _ = RAW[name]
    return f, float(wcs.wcs.crval[0]), float(wcs.wcs.crval[1]), pixscale(name)


def footprint(name):
    """[RA, Dec, radius] in degrees: a circle around the center of a loaded layer that covers its corners (for SIMBAD)."""
    data, w, _, _ = RAW[name]
    ny, nx = data.shape
    ra, dec = (float(v) for v in w.pixel_to_world_values((nx - 1) / 2, (ny - 1) / 2))
    cra, cdec = w.pixel_to_world_values(np.array([-0.5, nx - 0.5, -0.5, nx - 0.5]), np.array([-0.5, -0.5, ny - 0.5, ny - 0.5]))
    r0, d0, r1, d1 = np.radians(ra), np.radians(dec), np.radians(cra), np.radians(cdec)
    sep = np.degrees(np.arccos(np.clip(np.sin(d0) * np.sin(d1) + np.cos(d0) * np.cos(d1) * np.cos(r1 - r0), -1, 1)))
    return [ra, dec, float(np.max(sep))]


def sky_to_pix(name, ra, dec):
    """Pixel x, y (0-based, FITS orientation) on the grid of a loaded layer; NaN outside the image or its data."""
    if hasattr(ra, 'to_py'):
        ra, dec = ra.to_py(), dec.to_py()
    data, w, _, _ = RAW[name]
    ny, nx = data.shape
    x, y = w.world_to_pixel_values(np.asarray(ra, float), np.asarray(dec, float))
    x, y = np.atleast_1d(x).astype(float), np.atleast_1d(y).astype(float)
    ix, iy = np.round(x), np.round(y)
    inside = np.isfinite(x) & np.isfinite(y) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ok = np.zeros(len(x), bool)
    ok[inside] = np.isfinite(data[iy[inside].astype(int), ix[inside].astype(int)])
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
        data, text, f = read_sci(name, max_mp)
        _add(name, data, text, f, fill, cosmic)
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
