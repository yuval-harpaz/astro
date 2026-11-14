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
import time
import urllib.request
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from scipy.ndimage import map_coordinates, maximum_filter, label

warnings.simplefilter('ignore', FITSFixedWarning)

S3 = 'https://stpubdata.s3.amazonaws.com/jwst/public/'
RAW = {}     # name -> (data float32 2D, WCS), downsampled and hole-filled
REPROJ = {}  # (name, ref) -> float32 2D on ref grid
_T0 = [time.time()]


def reset_clock():
    _T0[0] = time.time()


def step(msg):
    print(f'[{time.time() - _T0[0]:6.1f}s] {msg}', flush=True)


def short(name):
    """jw07321-o004_t002_nircam_clear-f444w_i2d.fits -> F444W"""
    for part in name.replace('-', '_').split('_'):
        if len(part) > 3 and part[0] == 'f' and part[1:4].isdigit():
            return part.upper()
    return name


# ---------- stretch, copied from bot_grabber.py ----------
def to1(x):
    return 0.5 + (2.0 * x - 2.0) / (2 * np.sqrt((2.0 * x - 2.0) * (2.0 * x - 2.0) + 1.0))


def expand_highs(x):
    return np.piecewise(x, [x <= 0.9, x > 0.9],
                        [lambda x: x * 0.8 / 0.9, lambda x: 100.0 / 9.0 * (x - 0.9) ** 2 + 0.8 * x / 0.9])


def image_histogram_equalization(image, number_bins=10000, ignore0=False):
    shape = image.shape
    image = image.flatten()
    if ignore0:
        include = image > 0
    else:
        include = np.ones(len(image), bool)
    image_histogram, bins = np.histogram(image[include], number_bins, density=True)
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


def level_adjust(fits_arr, negative='consider', lims=[0.03, 0.98], factor=4.0, ignore0=False):
    fits_arr, mask = nanmask(fits_arr)
    hist_dat = fits_arr.flatten()
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
    if lims == [0.03, 0.98]:  # for backward compatibility
        rescaled = (fits_arr - minval) / (maxval - minval)
        rescaled_no_outliers = np.maximum(rescaled, np.quantile(rescaled, 0.002))
        rescaled_no_outliers = np.minimum(rescaled_no_outliers, np.quantile(rescaled_no_outliers, 1.0 - 0.002))
    else:
        no_outliers = np.maximum(fits_arr, minval)
        no_outliers = np.minimum(no_outliers, maxval)
        rescaled_no_outliers = (no_outliers - minval) / (maxval - minval)
        rescaled = rescaled_no_outliers
    if factor is None:
        adjusted = rescaled
    else:
        img_eqd = image_histogram_equalization(rescaled_no_outliers, ignore0=ignore0)
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


def block_factor(nx, ny, max_mp):
    return 1 if not max_mp else max(1, int(np.ceil(np.sqrt(nx * ny / (max_mp * 1e6)))))


def read_sci(fname, max_mp):
    """Download only the SCI extension and block-average it (nan-mean). Returns data, header text, factor."""
    url = s3url(fname)
    off = 0
    for ext in range(6):
        text, hdr, start, size = _read_header(url, off)
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


def add_layer(name, buf, ny, nx, header_text, factor, fill=True):
    """buf: bytes-like (or Pyodide JsProxy of a Uint8Array) of native float32, row-major (ny, nx)."""
    if hasattr(buf, 'to_py'):
        buf = buf.to_py()
    data = np.frombuffer(buf, dtype=np.float32).reshape(int(ny), int(nx)).copy()
    _add(name, data, header_text, int(factor), fill)
    return data.nbytes


def _add(name, data, header_text, factor, fill):
    if fill:
        t = time.time()
        data, n, it = fill_holes(data)
        step(f'{short(name)}: fill holes, {n} hole pixels, {it} iterations, {time.time() - t:.1f}s')
    RAW[name] = (data, binned_wcs(header_text, factor))


def has_layer(name):
    return name in RAW


def reproject_to(name, ref, chunk=256):
    key = (name, ref)
    if key in REPROJ:
        step(f'{short(name)}: reprojected layer cached')
        return REPROJ[key]
    data, wcs = RAW[name]
    if name == ref:
        out = data
    else:
        t = time.time()
        ref_data, ref_wcs = RAW[ref]
        ny, nx = ref_data.shape
        out = np.empty((ny, nx), np.float32)
        xx = np.arange(nx, dtype=np.float64)
        for y0 in range(0, ny, chunk):
            y1 = min(ny, y0 + chunk)
            gx, gy = np.meshgrid(xx, np.arange(y0, y1, dtype=np.float64))
            lon, lat = ref_wcs.pixel_to_world_values(gx, gy)
            px, py = wcs.world_to_pixel_values(lon, lat)
            out[y0:y1] = map_coordinates(data, [py, px], order=1, cval=np.nan, prefilter=False)
        step(f'{short(name)}: reprojected onto {short(ref)} grid ({nx}x{ny}), {time.time() - t:.1f}s')
    REPROJ[key] = out
    return out


def clear():
    RAW.clear()
    REPROJ.clear()
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


def compose_rgb(red, green, blue, factor=2.0, grey=True):
    """Returns uint8 (ny, nx, 3) with FITS orientation (origin lower-left). Reference = first layer listed."""
    red, green, blue = list(red), list(green), list(blue)
    allnames = red + green + blue
    if not allnames:
        raise ValueError('no layers chosen')
    ref = allnames[0]
    ny, nx = RAW[ref][0].shape
    rgb = np.zeros((ny, nx, 3), np.float32)
    for ich, names in enumerate([red, green, blue]):
        if not names:
            continue
        acc = np.zeros((ny, nx), np.float32)
        for name in names:
            lay = reproject_to(name, ref).copy()
            t = time.time()
            acc += level_adjust(lay, factor=factor)
            step(f'{short(name)}: level_adjust(factor={factor:g}), {time.time() - t:.1f}s')
        rgb[..., ich] = acc / len(names)
        step(f'{"RGB"[ich]} = ' + (f'mean of {len(names)} layers' if len(names) > 1 else short(names[0])))
    if not green and red and blue:  # like the bot with 2 files
        rgb[..., 1] = (rgb[..., 0] + rgb[..., 2]) / 2
        step('G = (R + B) / 2')
    rgb[np.isnan(rgb)] = 0
    if grey:
        rgb = grey_zeros(rgb, replace=np.max)
        step('grey_zeros: fill zero pixels of each channel with max of the other two')
    out = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    del rgb
    gc.collect()
    step(f'RGB image {nx}x{ny} ready')
    return out


def compose(red, green, blue, factor=2.0, grey=True):
    """For the browser: returns (rgba bytes flipped for canvas, height, width)."""
    rgb = compose_rgb(red, green, blue, factor, grey)
    ny, nx = rgb.shape[:2]
    rgba = np.empty((ny, nx, 4), np.uint8)
    rgba[..., :3] = rgb[::-1]  # FITS origin is lower-left, canvas is upper-left
    rgba[..., 3] = 255
    return rgba.tobytes(), ny, nx


def run_local(red, green, blue, max_mp=0, fill=True, grey=True, factor=2.0, out='color.jpg'):
    """Stand-alone: download, process and save a jpg. max_mp=0 means full resolution."""
    reset_clock()
    for name in dict.fromkeys(list(red) + list(green) + list(blue)):
        data, text, f = read_sci(name, max_mp)
        _add(name, data, text, f, fill)
    rgb = compose_rgb(red, green, blue, factor, grey)
    from PIL import Image
    Image.fromarray(rgb[::-1]).save(out, quality=95)
    step(f'saved {out}')
