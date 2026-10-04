"""
Light echo pipeline for JWST program 8527, thermal light echoes of the Cas A supernova.
https://www.stsci.edu/jwst-program-info/download/jwst/pdf/8527/

Targets:
    repeated: Greebo, Luna, Maki. 15 epochs spaced 14 days apart, plus an extra epoch "6point5". epoch 1 is the
              single-visit target (e.g. Greebo, o028), the rest are TARGET-Repeated (e.g. Greebo-Repeated, o017-o024,
              o047-o052). the time steps are what makes a 3D reconstruction possible.
    single:   Baby_Cas_A, Chloe, Coriander, Daphne, Juniper, Mab, Oliver. one visit, images only.
Every visit has F444W (the science filter) and one short filter, F070W or F200W.

Run without arguments to list options and targets.
Steps:
    download: download all public i2d files of TARGET and TARGET-Repeated that are missing locally
    images:   one RGB image per visit with a fixed stretch (saved to params.json on the first run),
              'full' on each visit's own grid, and 'aligned' after reprojection + registration to the
              reference visit, cropped to the reference view. aligned layers are cached as FITS for later steps.
    echo:     F444W visits colored by time lag as auto_plot 'filt' does by wavelength, red (first) to blue (last).
    volume:   3D dust volume (NIfTI, voxels in mpc). each visit is a slice at its light echo depth, computed from
              the echo paraboloid around Cas A, interpolated along the line of sight. repeated targets.
    view:     rotate the volume with pyvista (the MNE 3D backend). not in the default steps.
Existing images are skipped, use --overwrite to regenerate them.
"""
import sys
import re
import json
import argparse
from astro_utils import *
from bot_grabber import to1, expand_highs
from download_target import download_target_by_name
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion, shift as nd_shift
from skimage.registration import phase_cross_correlation
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', FITSFixedWarning)

PROGRAM = '8527'
STEPS = ['download', 'images', 'echo', 'sources', 'volume', 'view']
out_root = drive + 'light_echo/'
data_root = drive + 'data/'


def query_program(program=PROGRAM):
    """ all level 3 images of the program, public or not, one row per file """
    table = Observations.query_criteria(obs_collection='JWST', proposal_id=program, calib_level=3,
                                        dataproduct_type='image').to_pandas()
    table = table[table['intentType'] == 'science'].reset_index(drop=True)
    table['file'] = [u.split('/')[-1] for u in table['dataURL']]
    table['obs'] = [re.search(r'-(o\d+)_', f).group(1) for f in table['file']]
    table['base'] = [t.replace('-Repeated', '') for t in table['target_name']]
    table['public'] = table['dataRights'] == 'PUBLIC'
    for col in ['t_min', 't_obs_release']:
        table[col + '_iso'] = [Time(t, format='mjd').utc.iso[:10] for t in table[col]]
    return table


def get_table(program=PROGRAM):
    """ query MAST, fall back to the cached csv when offline """
    cache = out_root + f'jw0{program}_products.csv'
    try:
        table = query_program(program)
        if os.path.isdir(drive):
            os.makedirs(out_root, exist_ok=True)
            table.to_csv(cache, index=False)
    except Exception as err:
        if not os.path.isfile(cache):
            raise err
        print(f'MAST query failed ({err}), using {cache}')
        table = pd.read_csv(cache)
    return table


def is_local(row):
    fn = data_root + row['target_name'] + '/' + row['file']
    return os.path.isfile(fn) and os.path.getsize(fn) > 0


def list_targets(table):
    table = table.copy()
    table['local'] = [is_local(row) for _, row in table.iterrows()]
    repeated = set(table['base'][table['target_name'].str.endswith('-Repeated')])
    print(f'Targets in JWST program {PROGRAM} (visits: downloaded / public / planned):')
    for title, is_rep in [('repeated, TARGET + TARGET-Repeated (steps: download, images, echo)', True),
                          ('single visit (steps: download, images)', False)]:
        print(f'  {title}')
        for base, df in table.groupby('base'):
            if (base in repeated) != is_rep:
                continue
            visits = df.groupby('obs').agg(public=('public', 'all'), local=('local', 'all'),
                                           release=('t_obs_release_iso', 'max'))
            upcoming = visits['release'][~visits['public']]
            nxt = f'   next release {upcoming.min()}' if len(upcoming) else ''
            filts = sorted(set(re.search(r'clear-(f\d+\w)_', f).group(1) for f in df['file']))
            print(f'    {base:<12} {visits["local"].sum():>2} / {visits["public"].sum():>2} / {len(visits):>2}'
                  f'   {",".join(filts):<18}{nxt}')


def resolve_target(name, table):
    bases = sorted(table['base'].unique())
    match = [b for b in bases if b.lower() == name.replace('-Repeated', '').lower()]
    if len(match) == 0:
        raise SystemExit(f'unknown target {name}, choose from: {" ".join(bases)}')
    return match[0]


def download(base, table):
    df = table[(table['base'] == base) & table['public']]
    for tname in sorted(df['target_name'].unique()):
        dft = df[df['target_name'] == tname]
        have = sum(is_local(row) for _, row in dft.iterrows())
        print(f'{tname}: {have} of {len(dft)} public files already in {data_root + tname}')
        if have < len(dft):
            download_target_by_name(tname, date='all', proposal_id=PROGRAM)


## images

def filt_of(path):
    return re.search(r'clear-(f\d+\w)_', os.path.basename(path)).group(1)


def local_epochs(base):
    """ one row per visit with red (longest wavelength) and blue (shortest) filter files, sorted by time """
    files = sorted(glob(data_root + base + '/*_i2d.fits') + glob(data_root + base + '-Repeated/*_i2d.fits'))
    rows = []
    for f in files:
        h0 = fits.getheader(f, 0)
        rows.append({'obs': re.search(r'-(o\d+)_', f).group(1), 'filt': filt_of(f), 'file': f,
                     'mjd': h0['EXPMID'], 'date': h0['DATE-BEG'][:10]})
    df = pd.DataFrame(rows)
    epochs = []
    for obs, dfo in df.groupby('obs'):
        if len(dfo) < 2:
            print(f'skipping {obs}, only {dfo["filt"].values}')
            continue
        dfo = dfo.sort_values('filt', key=lambda x: [int(re.sub(r'\D', '', s)) for s in x])
        epochs.append({'obs': obs, 'date': dfo['date'].iloc[0], 'mjd': dfo['mjd'].mean(),
                       'blue': dfo['file'].iloc[0], 'red': dfo['file'].iloc[-1],
                       'blue_filt': dfo['filt'].iloc[0], 'red_filt': dfo['filt'].iloc[-1]})
    return pd.DataFrame(epochs).sort_values('mjd').reset_index(drop=True)


def fit_stretch(img, lims=(0.03, 0.995), stretch='asinh', asinh_k=10, factor=2, nbins=1024, sky_out=None):
    """
    freeze the stretch parameters on one visit so the same mapping applies to all visits.
    minval, maxval: quantiles lims of img, saved as values.
    stretch: 'linear' - clip((x - minval) / (maxval - minval), 0, 1)
             'asinh' - asinh(asinh_k * linear) / asinh(asinh_k), a fixed curve lifting faint values
             'level_adjust' - level_adjust math with the histogram equalization cdf of img frozen
    sky_out: None | float
        instead of the lower quantile, set minval so the median (sky) is displayed at sky_out (0 to 1).
        filters differ in dynamic range, so equal quantiles leave the sky brighter in some filters (F070W haze).
    """
    vals = img[np.isfinite(img) & (img != 0)]
    minval, maxval = np.quantile(vals, lims)
    if sky_out is not None and stretch != 'level_adjust':
        # invert the stretch at sky_out to get the linear fraction r, then solve (median - min) / (max - min) = r
        r = np.sinh(sky_out * np.arcsinh(asinh_k)) / asinh_k if stretch == 'asinh' else sky_out
        sky = np.median(vals)
        minval = (sky - r * maxval) / (1 - r)
    prm = {'stretch': stretch, 'minval': float(minval), 'maxval': float(maxval), 'lims': list(lims)}
    if sky_out is not None:
        prm['sky_out'] = sky_out
    if stretch == 'asinh':
        prm['asinh_k'] = asinh_k
    elif stretch == 'level_adjust':
        rescaled = np.clip((vals - minval) / (maxval - minval), 0, 1)
        hist, _ = np.histogram(rescaled, nbins, range=(0, 1))
        cdf = np.cumsum(hist) / np.sum(hist)
        prm['factor'] = factor
        prm['cdf'] = [round(float(c), 6) for c in cdf]
    elif stretch != 'linear':
        raise ValueError(f'unknown stretch {stretch}')
    return prm


def apply_stretch(img, prm):
    """ apply a frozen stretch from fit_stretch, returns values between 0 and 1, 0 where there is no data """
    mask = ~np.isfinite(img) | (img == 0)
    rescaled = np.clip((np.nan_to_num(img) - prm['minval']) / (prm['maxval'] - prm['minval']), 0, 1)
    stretch = prm.get('stretch', 'level_adjust')  # params.json from before 'stretch' was added
    if stretch == 'linear':
        adjusted = rescaled
    elif stretch == 'asinh':
        adjusted = np.arcsinh(prm['asinh_k'] * rescaled) / np.arcsinh(prm['asinh_k'])
    else:
        cdf = np.asarray(prm['cdf'])
        bins = np.linspace(0, 1, len(cdf) + 1)[:-1]
        eqd = np.interp(rescaled, bins, cdf)
        fac = prm['factor']
        eqd = (eqd ** fac + eqd ** (fac * 2) + eqd ** (fac * 4)) / 3.0
        adjusted = expand_highs((eqd + to1(rescaled)) * 0.5)
    adjusted = np.clip(adjusted, 0, 1)
    adjusted[mask] = 0
    return adjusted


def make_rgb(red, blue, prm_red, prm_blue):
    """ two layer RGB as in auto_plot: red, mean, blue, then whiten """
    r = apply_stretch(red, prm_red)
    b = apply_stretch(blue, prm_blue)
    rgb = np.stack([r, (r + b) / 2, b], axis=2)
    rgb = whiten_image(rgb)
    return (rgb * 255).astype('uint8')


def save_jpg(fn, rgb):
    plt.imsave(fn, rgb, origin='lower', pil_kwargs={'quality': 95})
    print(f'saved {fn}')


def star_image(img):
    """ high-pass and clip, leaving mostly point sources, for registration """
    img = np.nan_to_num(img)
    hp = img - gaussian_filter(img, 10)
    return np.clip(hp, 0, np.percentile(hp, 99.9))


def register(ref_blue, blue, max_shift=5):
    """ residual shift (dy, dx) of blue relative to ref_blue after WCS reprojection, using stars """
    overlap = np.isfinite(ref_blue) & np.isfinite(blue)
    a = star_image(np.where(overlap, ref_blue, 0))
    b = star_image(np.where(overlap, blue, 0))
    shift, error, _ = phase_cross_correlation(a, b, upsample_factor=20)
    if np.max(np.abs(shift)) > max_shift:
        print(f'WARNING: registration shift {shift} larger than {max_shift} pixels, not applied')
        shift = np.zeros(2)
    return shift


def shift_layer(img, shift):
    if not np.any(shift):
        return img
    return nd_shift(img, shift, order=1, mode='constant', cval=np.nan)


def read_layer(path, fill=True):
    """ i2d data and header. fill: fill holes (e.g. saturated star cores) with hole_func_fill, as auto_plot """
    with fits.open(path) as hdul:
        data = hdul[1].data.astype('float32')
        hdr = hdul[1].header
    if fill:
        data = hole_func_fill(data, func='max')
    return data, hdr


def aligned_layers(ep, ref, out, do_register=True, redo=False, fill=True):
    """ reproject both filters of a visit to the reference red grid, register by stars, cache as FITS """
    fns = {c: f"{out}aligned/{ep['obs']}_{ep[c + '_filt']}.fits" for c in ['red', 'blue']}
    if not redo and all(os.path.isfile(f) for f in fns.values()):
        hdrs = {c: fits.getheader(fns[c], 1) for c in fns}
        if all(h.get('REFOBS') == ref['obs'] and h.get('REGISTER') == do_register and h.get('FILLED', False) == fill
               for h in hdrs.values()):
            return {c: fits.getdata(fns[c], 1) for c in fns}
    ref_hdr = fits.getheader(ref['red'], 1)
    layers = {}
    for c in ['red', 'blue']:
        data, hdr = read_layer(ep[c], fill)
        if ep['obs'] == ref['obs'] and c == 'red':
            layers[c] = data
        else:
            layers[c], _ = reproject_interp((data, hdr), ref_hdr)
    shift = np.zeros(2)
    if do_register and ep['obs'] != ref['obs']:
        ref_blue = aligned_layers(ref, ref, out, do_register, redo=False, fill=fill)['blue']
        shift = register(ref_blue, layers['blue'])
        print(f"{ep['obs']} registration shift (dy, dx) = {np.round(shift, 2)} pixels")
    wcs_hdr = WCS(ref_hdr).to_header()
    for c in ['red', 'blue']:
        layers[c] = shift_layer(layers[c], shift).astype('float32')
        hdr = wcs_hdr.copy()
        hdr['BUNIT'] = 'MJy/sr'
        hdr['FILTER'] = ep[c + '_filt'].upper()
        hdr['EXPMID'] = ep['mjd']
        hdr['DATE-BEG'] = ep['date']
        hdr['REFOBS'] = ref['obs']
        hdr['REGISTER'] = do_register
        hdr['FILLED'] = fill
        hdr['SHIFTY'] = float(shift[0])
        hdr['SHIFTX'] = float(shift[1])
        hdr['SRCFILE'] = os.path.basename(ep[c])
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(layers[c], hdr)]).writeto(fns[c], overwrite=True)
    return layers


def crop_box(img):
    """ bounding box [y1, y2, x1, x2] of valid data """
    valid = np.isfinite(img) & (img != 0)
    ys, xs = np.where(valid)
    return [int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1]


def setup(base, ref_obs=None, reset_params=False):
    """ local visits, output folder, params.json content and the reference visit """
    epochs = local_epochs(base)
    if len(epochs) == 0:
        raise SystemExit(f'no local data for {base}, run the download step')
    print(epochs[['obs', 'date', 'blue_filt', 'red_filt']].to_string())
    out = out_root + base + '/'
    for sub in ['full', 'aligned', 'echo']:
        os.makedirs(out + sub, exist_ok=True)
    epochs.to_csv(out + 'epochs.csv', index=False)
    params_file = out + 'params.json'
    params = {}
    if os.path.isfile(params_file) and not reset_params:
        with open(params_file) as f:
            params = json.load(f)
        if ref_obs is not None and ref_obs != params['ref_obs']:
            raise SystemExit(f"{params_file} was made with ref {params['ref_obs']}, use --reset-params")
    if ref_obs is None:
        ref_obs = params.get('ref_obs', epochs['obs'].iloc[0])
    ref = epochs[epochs['obs'] == ref_obs]
    if len(ref) == 0:
        raise SystemExit(f'reference visit {ref_obs} not found locally')
    ref = ref.iloc[0]
    if not params:
        params = {'ref_obs': ref_obs, 'ref_date': ref['date'], 'filters': {}}
    return epochs, out, params, ref


def save_params(params, out):
    with open(out + 'params.json', 'w') as f:
        json.dump(params, f, indent=1)
    print(f'saved parameters to {out}params.json')


def images(base, ref_obs=None, lims=(0.03, 0.995), stretch='asinh', asinh_k=10, factor=2, blue_sky=0.04,
           reset_params=False, do_register=True, redo=False, overwrite=False, fill=True):
    """
    stretch parameters are fitted on the reference visit (and for filters it lacks, on the first visit that has
    them) and saved to params.json. the black point of blue filters puts their median sky at blue_sky.
    existing images are skipped unless overwrite (redo implies overwrite)
    """
    overwrite = overwrite or redo
    epochs, out, params, ref = setup(base, ref_obs, reset_params)
    if 'crop' not in params:
        params['crop'] = crop_box(aligned_layers(ref, ref, out, do_register=do_register, fill=fill)['red'])
        save_params(params, out)
    order = [ref.name] + [i for i in epochs.index if i != ref.name]
    for _, ep in epochs.loc[order].iterrows():
        name = f"{base}_{ep['date']}_{ep['obs']}"
        fn_aligned = f'{out}aligned/{name}_aligned.jpg'
        fn_full = f'{out}full/{name}.jpg'
        need_fit = any(ep[c + '_filt'] not in params['filters'] for c in ['red', 'blue'])
        do_aligned = overwrite or not os.path.isfile(fn_aligned)
        do_full = overwrite or not os.path.isfile(fn_full)
        if not (need_fit or do_aligned or do_full):
            print(f'{name} images exist, skipping')
            continue
        layers = aligned_layers(ep, ref, out, do_register=do_register, redo=redo, fill=fill)
        if need_fit:
            for c in ['red', 'blue']:
                filt = ep[c + '_filt']
                if filt not in params['filters']:
                    y1, y2, x1, x2 = params['crop']
                    params['filters'][filt] = fit_stretch(layers[c][y1:y2, x1:x2], lims=lims, stretch=stretch,
                                                          asinh_k=asinh_k, factor=factor,
                                                          sky_out=blue_sky if c == 'blue' else None)
                    params['filters'][filt]['source_obs'] = ep['obs']
            save_params(params, out)
        prm_red = params['filters'][ep['red_filt']]
        prm_blue = params['filters'][ep['blue_filt']]
        # aligned, cropped to the reference view
        if do_aligned:
            y1, y2, x1, x2 = params['crop']
            rgb = make_rgb(layers['red'][y1:y2, x1:x2], layers['blue'][y1:y2, x1:x2], prm_red, prm_blue)
            save_jpg(fn_aligned, rgb)
        del layers
        # full, on the visit's own red grid
        if not do_full:
            continue
        red, red_hdr = read_layer(ep['red'], fill)
        blue, _ = reproject_interp(read_layer(ep['blue'], fill), red_hdr)
        save_jpg(fn_full, make_rgb(red, blue, prm_red, prm_blue))


## echo: f444w visits colored by time, red (old) to blue (new)

def lag_colors(mjd, hue_max=0.68, uniform=True):
    """
    hue by time lag, as assign_colors_by_filt does by filter wavelength (without subtract_blue):
    norm = lag / max lag, the first visit is red (hue 0), the last is blue (hue_max).
    uniform: space the hsv hues by perceived color difference (CIEDE2000) instead of linearly. linear hsv spends a
    wide band of hues on similar greens, e.g. lags of 30 and 41 days out of 71 look alike (delta E 5 vs 28)
    """
    lag = np.asarray(mjd) - np.min(mjd)
    norm = lag / np.max(lag)
    if uniform:
        from skimage.color import rgb2lab, deltaE_ciede2000
        hues = np.linspace(0, hue_max, 1001)
        lab = rgb2lab(matplotlib.cm.hsv(hues)[:, :3][None])[0]
        cum = np.concatenate([[0], np.cumsum(deltaE_ciede2000(lab[:-1], lab[1:]))])
        hue = np.interp(norm, cum / cum[-1], hues)
    else:
        hue = norm * hue_max
    return matplotlib.cm.hsv(hue)[:, :3], lag


def time_composite(stack, colors, mode='sum'):
    """
    stack: (n, y, x) stretched layers, NaN where a visit has no data. colors: (n, 3)
    'sum': layer * color added over visits as in assign_colors. instead of dividing by the data max and blc,
           each channel is divided by the sum of the colors in that channel (a fixed white balance), so what is
           the same in all visits (stars, static dust) is white and only changes are colored.
    'mean': each channel is the color-weighted mean over visits covering the pixel.
    'max': each channel is the max over visits of layer * color, a single bright echo keeps its full color.
    no normalization by the data, the same visits and stretch always give the same image.
    """
    rgb = np.zeros(stack.shape[1:] + (3,), 'float32')
    if mode == 'sum':
        for lay, col in zip(stack, colors):
            rgb += np.nan_to_num(lay)[..., None] * col
        rgb /= np.sum(colors, axis=0)
    elif mode == 'mean':
        weight = np.zeros_like(rgb)
        for lay, col in zip(stack, colors):
            rgb += np.nan_to_num(lay)[..., None] * col
            weight += np.isfinite(lay)[..., None] * col
        rgb = np.where(weight > 1e-3, rgb / np.maximum(weight, 1e-3), 0)
    elif mode == 'max':
        for lay, col in zip(stack, colors):
            rgb = np.fmax(rgb, np.nan_to_num(lay)[..., None] * col)
    else:
        raise ValueError(f'unknown mode {mode}')
    return (np.clip(rgb, 0, 1) * 255).astype('uint8')


def draw_legend(rgb, colors, labels):
    """
    colored squares with labels at the bottom center (as the filt05 legend of auto_plot, which is bottom left).
    rgb is saved with origin='lower', so row 0 is the bottom of the image.
    """
    from cv2 import putText, getTextSize, FONT_HERSHEY_SIMPLEX, LINE_AA
    img = np.ascontiguousarray(rgb[::-1])  # display orientation
    n = len(labels)
    square = int(img.shape[0] / 10 / n)  # as in assign_colors_by_filt
    thickness = max(1, int(square / 8))
    (_, fh), _ = getTextSize('', FONT_HERSHEY_SIMPLEX, 100, thickness)
    font_scale = (square - 4 - thickness) / ((fh - 1) / 100)
    text_w = max(getTextSize(txt, FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0] for txt in labels)
    x0 = int((img.shape[1] - square - 4 - text_w) / 2)
    for i, (col, txt) in enumerate(zip(colors, labels)):
        start = img.shape[0] - (n - i) * square - square // 2  # half a square above the bottom edge
        img[start:start + square, x0:x0 + square] = (np.asarray(col) * 255).astype('uint8')
        putText(img, txt, (x0 + square + 4, start + square - 2), FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                thickness, LINE_AA)
    return img[::-1]


def echo(base, ref_obs=None, mode='sum', do_register=True, overwrite=False, fill=True, changes=False):
    epochs, out, params, ref = setup(base, ref_obs)
    if len(epochs) < 2:
        print(f'echo needs repeated visits, {base} has {len(epochs)}')
        return
    filt = epochs['red_filt'].iloc[0]
    if any(epochs['red_filt'] != filt):
        raise SystemExit(f'expected one red filter, got {set(epochs["red_filt"])}')
    name = f"{out}echo/{base}_echo_{filt}_{mode}_to{epochs['date'].iloc[-1]}"
    if not overwrite and os.path.isfile(name + '.jpg') and (not changes or os.path.isfile(name + '_changes.jpg')):
        print(f'{name}.jpg exists, skipping')
        return
    if 'crop' not in params or filt not in params['filters']:
        raise SystemExit('run the images step first')
    y1, y2, x1, x2 = params['crop']
    stack = np.zeros((len(epochs), y2 - y1, x2 - x1), 'float32')
    for iep, ep in epochs.iterrows():
        red = aligned_layers(ep, ref, out, do_register=do_register, fill=fill)['red'][y1:y2, x1:x2]
        stack[iep] = apply_stretch(red, params['filters'][filt])
        stack[iep][~np.isfinite(red) | (red == 0)] = np.nan
    colors, lag = lag_colors(epochs['mjd'].values)
    labels = [f'{d[8:10]}/{d[5:7]}/{d[:4]}' for d in epochs['date']]  # DD/MM/YYYY
    save_jpg(name + '.jpg', draw_legend(time_composite(stack, colors, mode), colors, labels))
    if not changes:
        return
    # changes only: subtract the per-pixel minimum over visits, removing stars and static dust.
    # needs at least two visits covering the pixel
    enough = np.sum(np.isfinite(stack), axis=0) >= 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # All-NaN slice
        static = np.where(enough, np.nanmin(stack, axis=0), np.nan)
    save_jpg(name + '_changes.jpg', draw_legend(time_composite(stack - static, colors, mode), colors, labels))


## sources: stars and background galaxies, which stay the same through the visits

def sources(base, ref_obs=None, region=None, k=5, smooth=8, min_area=4, min_visits=2, var_thr=0.175, lc_thr=0.1,
            blue_red_min=0.3, axis_max=2.0, bright_star=5.0, grow_k=2, grow_smooth=32, grow_max=40, do_register=True,
            fill=True, starnet=False, starnet_star=0.5,
            star_k=10):
    """
    find compact sources in the sum of the visits and decide which are galaxies to remove.
    1. sum: each visit's F444W minus its sky (percentile 10), averaged over the visits covering each pixel.
    2. clusters: small scale structure (sum minus a gaussian smoothed version, sigma smooth) above k noise levels,
       covered by at least min_visits, connected, at least min_area pixels. some clusters are patches of dust.
    3. per cluster: variability, the mean over its pixels of std / mean over the visits; light curve range,
       (max - min) / mean of the cluster's mean value per visit (noise and registration jitter average out);
       blue / red, small scale peak in the mean blue filter over that in F444W; axis ratio from second moments.
    4. star: has a blue point source, blue / red >= blue_red_min (stars are bright in blue, galaxies red), and
       axis ratio <= axis_max or an F444W peak above bright_star MJy/sr (spikes make bright stars look elongated).
    5. galaxy: not a star, variability < var_thr and light curve range < lc_thr.
    6. removal mask: galaxies grown while the minimum over the visits covering a pixel, of each visit minus its
       gaussian smoothed version (sigma grow_smooth, so relative to the local dust), stays above grow_k single visit
       noise levels, i.e. until a pixel is low in any visit, at most grow_max pixels.
    saves to out/sources/: clusters image (png, interactive html), a histogram, a table, the sum with the removal
    mask set to black, and arrays for pixel_info.
    starnet: for the StarNet2 route (files with _starnet). statistics stay on the original visits (StarNet2 removes
    galaxies partly and differently per visit, which would make them look variable), but a star is a cluster whose
    light StarNet2 removed mostly (> starnet_star of it, mean over pixels and visits), replacing the blue tests that
    fail in F200W where galaxies are bright. galaxies: not stars and static, as above.
    """
    tag = '_starnet' if starnet else ''
    from scipy.ndimage import mean as label_mean, maximum as label_max, find_objects, maximum_position
    epochs, out, params, ref = setup(base, ref_obs)
    os.makedirs(out + 'sources', exist_ok=True)
    cy1, cy2, cx1, cx2 = params['crop']
    if region is None:
        region = right_region(aligned_layers(ref, ref, out, do_register=do_register, fill=fill)['red'][cy1:cy2, cx1:cx2])
    ry1, ry2, rx1, rx2 = region
    dust, blues, cores, noises, removed = [], [], [], [], []
    for _, ep in epochs.iterrows():
        lay = aligned_layers(ep, ref, out, do_register=do_register, fill=fill)
        red = lay['red'][cy1:cy2, cx1:cx2][ry1:ry2, rx1:rx2].astype('float32')
        blue = lay['blue'][cy1:cy2, cx1:cx2][ry1:ry2, rx1:rx2].astype('float32')
        valid = np.isfinite(red) & (red != 0)
        if starnet:  # fraction of each pixel's light removed by StarNet2
            removed.append(np.where(valid, (red - starnet_starless(out, ep['obs'], red, valid)) /
                                    np.maximum(red - np.percentile(red[valid], 10), 1e-6), np.nan))
        dust.append(np.where(valid, red - np.percentile(red[valid], 10), np.nan))
        noises.append(pixel_noise(np.where(valid, red, np.nan)))
        bvalid = np.isfinite(blue) & (blue != 0)
        blues.append(np.where(bvalid, blue - np.percentile(blue[bvalid], 10), np.nan))
        # point sources in the blue filter, as in star_mask
        bfill = np.where(bvalid, blue, np.median(blue[bvalid]))
        compact = bfill - gaussian_filter(bfill, 8)
        cores.append(bvalid & (compact > star_k * pixel_noise(np.where(bvalid, blue, np.nan))))
    dust = np.array(dust)
    ncov = np.sum(np.isfinite(dust), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # uncovered pixels
        total = np.nanmean(dust, axis=0)
        std_t = np.nanstd(dust, axis=0)
        # small scale structure per visit, for growing the removal mask relative to the local dust
        low_t = np.nanmin([np.where(np.isfinite(dv), dv - gaussian_filter(np.nan_to_num(dv), grow_smooth), np.nan)
                           for dv in dust], axis=0)
        blue_mean = np.nanmean(blues, axis=0)
    covered = ncov >= min_visits
    variab = np.where(covered & (total > 0), std_t / np.maximum(total, 1e-6), np.nan)
    filled = np.where(np.isfinite(total), total, 0)
    small = filled - gaussian_filter(filled, smooth)
    bfilled = np.nan_to_num(blue_mean)
    bsmall = bfilled - gaussian_filter(bfilled, smooth)
    noise = pixel_noise(np.where(np.isfinite(total), total, np.nan))
    labels, n = label((small > k * noise) & covered)
    area = np.bincount(labels.ravel(), minlength=n + 1)
    keep = area >= min_area
    keep[0] = False
    labels = np.where(keep[labels], labels, 0)
    ids = np.nonzero(keep)[0]
    objs = find_objects(labels)
    star_any = np.any(cores, axis=0)
    has_core = np.array(label_mean(star_any.astype(float), labels, ids)) > 0
    red_peak = np.array(label_max(small, labels, ids))
    blue_red = np.array(label_max(bsmall, labels, ids)) / np.maximum(red_peak, 1e-6)
    peak = np.array(maximum_position(filled, labels, ids))  # the brightest pixel, always inside the cluster
    cy, cx = peak[:, 0], peak[:, 1]
    cvar, lcr, axr = [], [], []
    for i in ids:
        sl = objs[i - 1]
        m = labels[sl] == i
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cvar.append(np.nanmean(variab[sl][m]))
            curve = np.nanmean(dust[:, sl[0], sl[1]][:, m], axis=1)
        curve = curve[np.isfinite(curve)]
        lcr.append((curve.max() - curve.min()) / max(abs(curve.mean()), 1e-6) if len(curve) > 1 else np.nan)
        yy, xx = np.nonzero(m)
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.vstack([xx, yy])))) if len(xx) > 2 else np.ones(2)
        axr.append(float(np.sqrt(ev[1] / max(ev[0], 1e-6))))
    table = pd.DataFrame({'id': ids, 'area_px': area[ids], 'y': cy, 'x': cx, 'mean_sum': label_mean(filled, labels, ids),
                          'peak': filled[cy, cx], 'variability': cvar, 'lc_range': lcr, 'blue_red': blue_red,
                          'axis_ratio': axr, 'blue_core': has_core})
    table['star'] = (table['blue_core'] & (table['blue_red'] >= blue_red_min) &
                     ((table['axis_ratio'] <= axis_max) | (table['peak'] > bright_star)))
    if starnet:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            frac = np.clip(np.nanmean(removed, axis=0), 0, 1)
        table['starnet_removed'] = np.array(label_mean(np.nan_to_num(frac), labels, ids))
        table['star'] = table['starnet_removed'] > starnet_star
    table['galaxy'] = ~table['star'] & (table['variability'] < var_thr) & (table['lc_range'] < lc_thr)
    # each visit's value at the cluster center (brightest pixel)
    for iv, (obs, date) in enumerate(zip(epochs['obs'], epochs['date'])):
        table[f'{obs}_{date}'] = dust[iv][cy, cx]
    # removal mask: grow galaxies while the minimum over the covering visits stays above grow_k noise levels
    seeds = np.isin(labels, table['id'][table['galaxy']])
    allowed = covered & (np.nan_to_num(low_t, nan=-1) > grow_k * max(noises))
    remove = binary_dilation(seeds, mask=allowed | seeds, iterations=grow_max) if seeds.any() else seeds
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        low_raw = np.nanmin(dust, axis=0)
    fill_m, void_m = split_fill_void(remove, total, variab, low_raw, noise)
    print(f'removal: {100 * fill_m[covered].mean():.2f}% filled (surrounded by dust), '
          f'{100 * void_m[covered].mean():.2f}% void (empty space, grown further)')
    table.to_csv(out + f'sources/{base}_sources{tag}.csv', index=False)
    print(f'{n} clusters above {k} noise levels, {len(ids)} with >= {min_area} px: {table["star"].sum()} stars, '
          f'{table["galaxy"].sum()} galaxies (variability < {var_thr}, light curve range < {lc_thr}), '
          f'{(~table["star"] & ~table["galaxy"]).sum()} other. removal mask {100 * remove[covered].mean():.2f}% '
          f'of the region (clusters alone {100 * seeds[covered].mean():.2f}%)')
    # arrays for looking up pixels later (region coordinates: x = column, y = row, origin at the bottom left)
    np.save(out + f'sources/{base}_layers{tag}.npy', dust.astype('float32'))  # (visit, y, x), sky subtracted F444W
    np.savez(out + f'sources/{base}_maps{tag}.npz', total=total.astype('float32'), variability=variab.astype('float32'),
             labels=labels.astype('int32'), stars=star_any, remove=remove, remove_fill=fill_m, remove_void=void_m)
    with open(out + f'sources/{base}_coords{tag}.json', 'w') as f:
        json.dump({'region_crop': [int(r) for r in region], 'crop': params['crop'],
                   'note': 'x, y are region pixels; reference grid pixel = crop start + region start + x (or y)',
                   'visits': list(epochs['obs']), 'dates': list(epochs['date']), 'var_thr': var_thr,
                   'lc_thr': lc_thr}, f, indent=1)
    sources_figures(base + tag, out, total, labels, table, var_thr, fill_m, void_m)
    sources_html(base + tag, out, total, labels, table, var_thr, fill_m, void_m)
    return table


def split_fill_void(remove, total, variab, low, noise, dust_k=5, dust_var=0.2, ring=(5, 20), far=(25, 40),
                    void_k=3, void_grow=40, void_pad=3, fill_pad=3):
    """
    removed objects surrounded by dust are filled from their surroundings, objects in empty space become void
    (sky). dust around an object: in a ring ring[0]..ring[1] pixels outside it, the mean image is above dust_k
    noise levels and varies between visits (median variability > dust_var); a galaxy's own halo is bright but
    static. void objects grow while low, the minimum over the visits smoothed by sigma 2 (a halo is in every visit,
    faint dust drops in some), is above its background (25th percentile in a ring far[0]..far[1] pixels out, beyond
    the halo, median) + void_k times its spread there, up to void_grow pixels, then void_pad more, so faint outskirts do not
    stay as hollow frames. filled objects grow fill_pad pixels.
    returns fill and void masks
    """
    from scipy.ndimage import find_objects
    fill, void = np.zeros(remove.shape, bool), np.zeros(remove.shape, bool)
    labels, n = label(remove)
    pad = max(far[1], void_grow) + void_pad + 2
    finite = np.isfinite(total)
    low_s = gaussian_filter(np.nan_to_num(low), 2)
    for i, sl in enumerate(find_objects(labels), 1):
        win = (slice(max(sl[0].start - pad, 0), sl[0].stop + pad), slice(max(sl[1].start - pad, 0), sl[1].stop + pad))
        reg = labels[win] == i
        near = (binary_dilation(reg, iterations=ring[1]) & ~binary_dilation(reg, iterations=ring[0]) &
                ~remove[win] & finite[win])
        dusty = (near.sum() > 10 and np.median(total[win][near]) > dust_k * noise and
                 np.nanmedian(variab[win][near]) > dust_var)
        if dusty:
            fill[win] |= binary_dilation(reg, iterations=fill_pad)
        else:
            out_ring = (binary_dilation(reg, iterations=far[1]) & ~binary_dilation(reg, iterations=far[0]) &
                        ~remove[win] & finite[win])
            # background level and spread of low in the far ring (neighbour differences underestimate the noise
            # of a smoothed image)
            vals = low_s[win][out_ring] if out_ring.sum() > 10 else np.zeros(1)
            bg = float(np.median(vals))
            spread = 1.4826 * float(np.median(np.abs(vals - bg))) + 1e-6
            grown = binary_dilation(reg, mask=reg | (low_s[win] > bg + void_k * spread), iterations=void_grow)
            void[win] |= binary_dilation(grown, iterations=void_pad)
    return fill & ~void, void


def pixel_info(base, x, y, starnet=False):
    """ what the sources step saw at region pixel x, y: the sky subtracted F444W of each visit, variability, cluster """
    out = out_root + base + '/sources/'
    tag = '_starnet' if starnet else ''
    layers = np.load(out + f'{base}_layers{tag}.npy', mmap_mode='r')
    maps = np.load(out + f'{base}_maps{tag}.npz')
    with open(out + f'{base}_coords{tag}.json') as f:
        coords = json.load(f)
    table = pd.read_csv(out + f'{base}_sources{tag}.csv')
    lab = int(maps['labels'][y, x])
    print(f'{base} x {x}, y {y} (reference grid x {coords["crop"][2] + coords["region_crop"][2] + x}, '
          f'y {coords["crop"][0] + coords["region_crop"][0] + y})')
    for obs, date, v in zip(coords['visits'], coords['dates'], layers[:, y, x]):
        print(f'  {obs} {date}: {v:8.4f} MJy/sr above sky')
    print(f"  mean {maps['total'][y, x]:.4f}, variability {maps['variability'][y, x]:.3f}, "
          f"star core in blue: {bool(maps['stars'][y, x])}")
    if lab:
        row = table[table['id'] == lab].iloc[0]
        kind = 'star' if row['star'] else 'galaxy' if row['galaxy'] else 'other'
        print(f"  cluster {lab}: {kind}, area {row['area_px']} px, variability {row['variability']:.3f}")
    else:
        print('  not in a cluster')


def sources_html(base, out, total, labels, table, var_thr, fill_m=None, void_m=None):
    """
    interactive version of the clusters image (plotly): hover shows x, y (region pixels, origin bottom left),
    and over a cluster center its id, area, variability and class
    """
    import plotly.graph_objects as go
    shown = np.nan_to_num(np.arcsinh(np.clip(total, 0, None) / 0.05)) / 3
    gray = (np.clip(shown, 0, 1) * 255).astype('uint8')
    rgb = np.stack([gray] * 3, axis=2)
    lut = np.full(labels.max() + 1, np.nan)
    lut[table['id']] = table['variability']
    varimg = np.where(labels > 0, lut[labels], np.nan)
    vmax = float(np.nanpercentile(table['variability'], 95))
    cols = (np.array(matplotlib.colormaps['viridis'](np.clip(np.nan_to_num(varimg) / vmax, 0, 1))[..., :3]) * 255)
    on = np.isfinite(varimg)
    rgb[on] = cols[on].astype('uint8')
    for m, color in [(fill_m, [255, 60, 200]), (void_m, [255, 160, 0])]:  # outlines of the removal masks
        if m is not None:
            rgb[m & ~binary_erosion(m)] = color
    fig = go.Figure()
    fig.add_trace(go.Image(z=rgb, x0=0, dx=1, y0=0, dy=1, name='',
                           hovertemplate='x %{x}, y %{y}<extra></extra>'))
    kind = np.where(table['star'], 'star', np.where(table['galaxy'], 'galaxy', 'other'))
    visit_cols = [c for c in table.columns if c[0] == 'o' and c[1:4].isdigit()]
    hover = []
    for _, row in table.iterrows():
        visits = '<br>'.join(f"{c[:4]} {c[5:]}: {row[c]:.4f}" for c in visit_cols)
        hover.append(f"cluster {row['id']}<br>{'star' if row['star'] else 'galaxy' if row['galaxy'] else 'other'}"
                     f"<br>x {row['x']:.0f}, y {row['y']:.0f}<br>area {row['area_px']} px"
                     f"<br>variability {row['variability']:.3f}, light curve range {row['lc_range']:.3f}"
                     f"<br>blue/red {row['blue_red']:.2f}, axis ratio {row['axis_ratio']:.1f}"
                     f"<br>MJy/sr above sky at this pixel:<br>{visits}")
    symbols = {'star': 'star', 'galaxy': 'circle', 'other': 'diamond'}
    for k in ['galaxy', 'star', 'other']:
        sel = kind == k
        fig.add_trace(go.Scatter(
            x=table['x'][sel], y=table['y'][sel], mode='markers', name=k,
            marker=dict(size=5, symbol=symbols[k], color=table['variability'][sel], colorscale='Viridis', cmin=0,
                        cmax=vmax, line=dict(width=0.5, color='white'),
                        colorbar=dict(title='variability', x=1.0) if k == 'galaxy' else None),
            text=np.array(hover)[sel], hovertemplate='%{text}<extra></extra>', visible='legendonly' if k == 'other' else True))
    ny, nx = total.shape
    fig.update_layout(
        title=f'{base}: clusters colored by variability. x, y are region pixels. removed galaxies: magenta = filled '
              f'(in dust), orange = void (empty space)',
        xaxis=dict(range=[0, nx], constrain='domain', title='x'),
        yaxis=dict(range=[0, ny], scaleanchor='x', title='y', autorange=None),
        height=950, width=1250, template='plotly_dark', legend=dict(title='cluster centers'))
    fn = out + f'sources/{base}_clusters.html'
    fig.write_html(fn, include_plotlyjs=True, full_html=True)
    print(f'saved {fn}')


def sources_figures(base, out, total, labels, table, var_thr, fill_m=None, void_m=None):
    """ clusters colored by variability, the variability histogram, and the sum with galaxies set to black """
    ink = '#333333'
    v = np.isfinite(total)
    shown = np.nan_to_num(np.arcsinh(np.clip(total, 0, None) / 0.05))
    # map variability onto the label image
    lut = np.full(labels.max() + 1, np.nan)
    lut[table['id']] = table['variability']
    varimg = np.where(labels > 0, lut[labels], np.nan)
    star_lab = np.isin(labels, table['id'][table['star']])
    if fill_m is None:
        fill_m = np.zeros(labels.shape, bool)
        void_m = np.isin(labels, table['id'][table['galaxy']])
    step = max(1, int(np.ceil(max(total.shape) / 1600)))
    sl = (slice(None, None, step), slice(None, None, step))
    fig, ax = plt.subplots(figsize=(13, 12 * total.shape[0] / total.shape[1]))
    ax.imshow(shown[sl], origin='lower', cmap='gray', vmin=0, vmax=3)
    vmax = np.nanpercentile(table['variability'], 95)
    im = ax.imshow(np.ma.masked_invalid(varimg[sl]), origin='lower', cmap='viridis', vmin=0, vmax=vmax,
                   interpolation='nearest')
    ax.contour(star_lab[sl], [0.5], colors='#ff4040', linewidths=0.5)
    ax.set_title(f'{base}: clusters in the sum of the visits, colored by variability (std / mean over visits); '
                 'red outline = star (blue filter)', fontsize=9, color=ink, loc='left')
    ax.axis('off')
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label('variability', fontsize=8, color=ink)
    cb.ax.tick_params(labelsize=7, colors=ink)
    fig.tight_layout()
    fn = out + f'sources/{base}_clusters.png'
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print(f'saved {fn}')
    # histogram
    fig, ax = plt.subplots(figsize=(7, 3.5))
    rng = (0, np.nanpercentile(table['variability'], 98))
    ax.hist(table['variability'][~table['star']].dropna(), bins=50, range=rng, color='#4a7ab0', label='not star')
    ax.hist(table['variability'][table['star']].dropna(), bins=50, range=rng, color='#d0603a', alpha=0.8,
            label='star (blue filter)')
    ax.axvline(var_thr, color=ink, lw=1, ls='--')
    ax.text(var_thr, ax.get_ylim()[1] * 0.95, f' variability < {var_thr:.3f} (and flat light curve)', fontsize=8,
            color=ink, va='top')
    ax.set_xlabel('cluster variability (mean over pixels of std / mean over visits)', fontsize=8, color=ink)
    ax.set_ylabel('clusters', fontsize=8, color=ink)
    ax.tick_params(labelsize=7, colors=ink)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fn = out + f'sources/{base}_variability_hist.png'
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print(f'saved {fn}')
    # galaxies set to black
    valid = np.isfinite(total)
    cleaned = fill_stars(np.where(valid, total, 0).astype('float32'), valid, fill_m)
    cleaned[void_m] = 0
    clean = np.nan_to_num(np.arcsinh(np.clip(cleaned, 0, None) / 0.05))
    fig, axs = plt.subplots(1, 2, figsize=(18, 9 * total.shape[0] / total.shape[1]))
    for a, img, title in [(axs[0], shown, 'sum of the visits'), (axs[1], clean, 'galaxies removed: filled in dust, void in empty space')]:
        a.imshow(img[sl], origin='lower', cmap='gray', vmin=0, vmax=3)
        a.set_title(title, fontsize=9, color=ink, loc='left')
        a.axis('off')
    fig.tight_layout()
    fn = out + f'sources/{base}_galaxies_black.png'
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print(f'saved {fn}')


## volume: light echoes as slices of the dust cloud

PC_PER_LY = 0.306601
CAS_A = SkyCoord('23h23m24s', '+58d48m54s')  # SIMBAD, J2000


def echo_depth(rho, age_yr):
    """
    light echo paraboloid, Cas A at the focus: z = rho^2 / (2ct) - ct / 2 (pc), z toward the observer.
    rho: distance from Cas A on the sky plane (pc). age_yr: time since the explosion light reached Earth.
    """
    ct = age_yr * PC_PER_LY
    return rho ** 2 / (2 * ct) - ct / 2


def pixel_noise(img):
    """ robust per-pixel noise from differences of neighbouring pixels """
    d = np.diff(img, axis=1)
    d = d[np.isfinite(d)]
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)


def block_mean(img, b):
    """ bin by b x b with nanmean, trimming edges """
    ny, nx = (img.shape[0] // b) * b, (img.shape[1] // b) * b
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN blocks
        return np.nanmean(img[:ny, :nx].reshape(ny // b, b, nx // b, b), axis=(1, 3))


def star_mask(blue, bvalid, red, rvalid, star_k=10, dust_k=5, star_dilate=1.5, star_max=25, star_power=0.25):
    """
    stars are found in the blue filter, where dust is faint, as point sources: above a smoothed version by star_k
    noise levels, so extended dust (visible in F200W) is not masked. each source is grown to cover its F444W halo,
    by star_dilate * (peak / threshold) ** star_power pixels up to star_max, where peak is the source's compact
    F444W brightness and threshold is dust_k F444W noise levels. sizing by F444W, not blue,
    keeps faint red stars that are deep in F200W from getting large masks
    """
    from scipy.ndimage import distance_transform_edt, maximum as label_max

    def compact(img, valid):
        filled = np.where(valid, img, np.median(img[valid]))
        return filled - gaussian_filter(filled, 8)

    cores = bvalid & (compact(blue, bvalid) > star_k * pixel_noise(np.where(bvalid, blue, np.nan)))
    labels, n = label(cores)
    if n == 0:
        return cores
    thr = dust_k * pixel_noise(np.where(rvalid, red, np.nan))
    peaks = np.asarray(label_max(compact(red, rvalid), labels, index=np.arange(1, n + 1)))
    radii = np.minimum(star_dilate * np.maximum(peaks / thr, 1) ** star_power, star_max)
    radius_img = np.zeros(blue.shape)
    radius_img[cores] = radii[labels[cores] - 1]
    mask = np.zeros(blue.shape, bool)
    edges = [0, 4, 8, 16, 32, 64, np.inf]
    for lo, hi in zip(edges[:-1], edges[1:]):  # radius classes, one distance transform each
        seed = cores & (radius_img > lo) & (radius_img <= hi)
        if seed.any():
            mask |= distance_transform_edt(~seed) <= min(hi, radius_img[seed].max())
    return mask


def right_region(red):
    """ region [y1, y2, x1, x2] (crop coordinates) right of the gap between the two NIRCam modules """
    cov = np.mean(np.isfinite(red) & (red != 0), axis=0)
    nx = len(cov)
    gap = int(0.3 * nx) + int(np.argmin(cov[int(0.3 * nx):int(0.7 * nx)]))
    return [0, red.shape[0], gap, nx]


def fill_stars(red, valid, stars):
    """
    fill masked stars from their surroundings, no holes left: normalized convolution, a gaussian weighted mean of
    the unmasked neighbours, from small to large scales until every masked pixel is filled
    """
    good = valid & ~stars
    img, weight = np.where(good, red, 0), good.astype(float)
    out = np.where(valid, red, np.nan)
    todo = stars & valid
    for sigma in [1, 2, 4, 8, 16, 32]:
        den = gaussian_filter(weight, sigma)
        fillable = todo & (den > 0.05)
        out[fillable] = gaussian_filter(img, sigma)[fillable] / den[fillable]
        todo &= ~fillable
        if not todo.any():
            break
    out[todo] = np.median(red[good])
    return out


def flow_interp(prev, cur, w, flow):
    """
    motion compensated interpolation between two visits at fraction w (0 = prev, 1 = cur), per pixel.
    flow (vy, vx) from optical_flow_tvl1(prev, cur): prev(x) ~ cur(x + v). features move along the flow instead of
    fading between two positions, which leaves a dim double image (looks like a gap) with linear interpolation
    """
    from scipy.ndimage import map_coordinates
    vy, vx = flow
    yy, xx = np.mgrid[0:prev.shape[0], 0:prev.shape[1]].astype('float32')
    out = []
    for img, s in [(prev, -w), (cur, 1 - w)]:
        cov = map_coordinates(np.isfinite(img).astype('float32'), [yy + s * vy, xx + s * vx], order=0, cval=0) > 0
        val = map_coordinates(np.nan_to_num(img), [yy + s * vy, xx + s * vx], order=1, cval=0)
        out.append((np.where(cov, val, 0), cov))
    (a, ca), (b, cb) = out
    blend = np.where(ca & cb, (1 - w) * a + w * b, np.where(ca, a, b))
    return np.where(ca | cb, blend, 0)


def sources_masks(base, ref_obs, region, do_register=True, fill=True, starnet=False):
    """ fill and void masks from the sources step, for the region; runs sources if missing or for another region """
    out = out_root + base + '/sources/'
    tag = '_starnet' if starnet else ''
    try:
        with open(out + f'{base}_coords{tag}.json') as f:
            same = json.load(f)['region_crop'] == [int(r) for r in region]
        maps = np.load(out + f'{base}_maps{tag}.npz')
        if not same or 'remove_void' not in maps:
            raise FileNotFoundError
    except (FileNotFoundError, KeyError):
        sources(base, ref_obs=ref_obs, region=region, do_register=do_register, fill=fill, starnet=starnet)
        maps = np.load(out + f'{base}_maps{tag}.npz')
    return maps['remove_fill'], maps['remove_void']


def starnet_starless(out, obs, red, valid, scale=20.0):
    """
    StarNet2 (CLI, installed separately) starless version of a visit's F444W, in the same units. cached in
    out/starnet/. the input is scaled by 1 / scale (StarNet clips float FITS above 1; only star cores exceed 20
    MJy/sr), uncovered pixels set to the median (StarNet rejects NaN), run with --linear (it stretches for the
    network and reverses the stretch). returns NaN outside valid.
    """
    import subprocess
    os.makedirs(out + 'starnet', exist_ok=True)
    fin, fout = out + f'starnet/{obs}_in.fits', out + f'starnet/{obs}_starless.fits'
    img = (np.where(valid, red, np.median(red[valid])) / scale).astype('float32')
    cached = os.path.isfile(fout) and os.path.isfile(fin) and fits.getdata(fin).shape == img.shape and \
        np.allclose(fits.getdata(fin), img)
    if not cached:
        fits.PrimaryHDU(img).writeto(fin, overwrite=True)
        print(f'StarNet2 on {obs} ...')
        subprocess.run(['starnet2', '--input', fin, '--output', fout, '--linear', '--quiet'], check=True,
                       stderr=subprocess.DEVNULL)
    return np.where(valid, fits.getdata(fout).astype('float32') * scale, np.nan)


def volume(base, ref_obs=None, region=None, binning=2, dust_k=5, star_k=10, star_dilate=1.5, star_max=25, sky_prc=10,
           distance_pc=3400, sn_year=1681.0, interp='flow', do_register=True, fill=True, overwrite=False,
           static=True, stars='mask'):
    """
    dust density proxy on a voxel grid. x, y: sky pixels (binned), depth: behind the first visit's echo surface.
    per visit: stars='mask': stars found in the blue filter are filled from their surroundings; stars='starnet':
    StarNet2's starless image is used instead (saved with _starnet). static: background galaxies from
    the sources step are filled when surrounded by dust and set to sky in empty space; sky (percentile sky_prc)
    subtracted, f444w binned. without static the volume is saved with _raw. each visit is placed at its echo depth,
    computed per pixel from the echo paraboloid,
    and the line of sight is interpolated between visits ('flow': motion compensated, 'linear', 'nearest').
    voxels are isotropic, in mpc, at the distance of the dust. values below 0 are 0; the dust threshold
    (dust_k noise levels) is saved in the json as the viewer's default.
    saves NIfTI (+ json) and a projections png to out/volume/.
    """
    import nibabel as nib
    from skimage.registration import optical_flow_tvl1
    epochs, out, params, ref = setup(base, ref_obs)
    if len(epochs) < 2:
        raise SystemExit(f'volume needs repeated visits, {base} has {len(epochs)}')
    if 'crop' not in params:
        raise SystemExit('run the images step first')
    os.makedirs(out + 'volume', exist_ok=True)
    name = (f"{out}volume/{base}_dust_bin{binning}_to{epochs['date'].iloc[-1]}" +
            ('_starnet' if stars == 'starnet' else '') + ('' if static else '_raw'))
    if not overwrite and os.path.isfile(name + '.nii.gz'):
        print(f'{name}.nii.gz exists, skipping')
        return name
    cy1, cy2, cx1, cx2 = params['crop']
    lay = aligned_layers(ref, ref, out, do_register=do_register, fill=fill)
    if region is None:
        region = right_region(lay['red'][cy1:cy2, cx1:cx2])
    ry1, ry2, rx1, rx2 = region
    print(f'region (crop coordinates) y {ry1}:{ry2}, x {rx1}:{rx2}')
    ref_hdr = fits.getheader(out + f"aligned/{ref['obs']}_{ref['red_filt']}.fits", 1)
    wcs = WCS(ref_hdr)
    pix_rad = np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * np.pi / 180
    # binned pixel centers in full reference grid coordinates -> angle from Cas A
    ny, nx = (ry2 - ry1) // binning, (rx2 - rx1) // binning
    yy, xx = np.mgrid[0:ny, 0:nx]
    xs = cx1 + rx1 + (xx + 0.5) * binning - 0.5
    ys = cy1 + ry1 + (yy + 0.5) * binning - 0.5
    theta = wcs.pixel_to_world(xs, ys).separation(CAS_A).rad
    # the dust is at distance_pc - z from us (z < 0 is behind Cas A), which sets rho and the pixel size
    age0 = Time(epochs['mjd'].iloc[0], format='mjd').decimalyear - sn_year
    z0 = np.zeros_like(theta)
    for _ in range(3):
        rho = (distance_pc - z0) * theta
        z0 = echo_depth(rho, age0)
    dust_dist = distance_pc - float(np.mean(z0))
    vox_mpc = pix_rad * dust_dist * 1000 * binning
    stack = np.zeros((len(epochs), ny, nx), 'float32')
    depth = np.zeros((len(epochs), ny, nx))  # mpc behind the first visit's surface
    reds, valids, starss, noises = [], [], [], []
    for iep, ep in epochs.iterrows():
        lay = aligned_layers(ep, ref, out, do_register=do_register, fill=fill)
        red = lay['red'][cy1:cy2, cx1:cx2][ry1:ry2, rx1:rx2].astype('float32')
        blue = lay['blue'][cy1:cy2, cx1:cx2][ry1:ry2, rx1:rx2].astype(float)
        valid = np.isfinite(red) & (red != 0)
        bvalid = np.isfinite(blue) & (blue != 0)
        reds.append(red)
        valids.append(valid)
        noises.append(pixel_noise(np.where(valid, red, np.nan)))
        if stars == 'starnet':
            reds[-1] = starnet_starless(out, ep['obs'], red, valid)
            starss.append(np.zeros(red.shape, bool))
        else:
            starss.append(star_mask(blue, bvalid, red, valid, star_k, dust_k, star_dilate, star_max))
    if static:
        fixed, void = sources_masks(base, ref_obs, region, do_register, fill, starnet=stars == 'starnet')
    else:
        fixed = void = np.zeros(reds[0].shape, bool)
    info = []
    for iep, ep in epochs.iterrows():
        red, valid, star_m, noise = reds[iep], valids[iep], starss[iep], noises[iep]
        sky = np.percentile(red[valid], sky_prc)
        cleaned = fill_stars(red, valid, star_m | fixed) if (star_m | fixed).any() else red.copy()
        cleaned[void & valid] = sky  # galaxies in empty space: sky
        dust = block_mean(cleaned - sky, binning)[:ny, :nx]
        stack[iep] = dust
        age = Time(ep['mjd'], format='mjd').decimalyear - sn_year
        depth[iep] = (z0 - echo_depth(rho, age)) * 1000
        thr = dust_k * noise / binning
        info.append({'obs': ep['obs'], 'date': ep['date'], 'age_yr': age, 'sky': float(sky), 'noise': noise,
                     'threshold_binned': thr, 'star_fraction': float(star_m[valid].mean()),
                     'galaxy_fill_fraction': float(fixed[valid].mean()),
                     'galaxy_void_fraction': float(void[valid].mean()), 'depth_mpc_mean': float(np.mean(depth[iep]))})
        print(f"{ep['obs']} {ep['date']} depth {np.mean(depth[iep]):6.2f} mpc, sky {sky:.3f}, noise {noise:.4f}, "
              f"stars filled {100 * star_m[valid].mean():.1f}%, above threshold {100 * np.nanmean(dust > thr):.1f}%")
    del reds, starss
    stack = np.where(np.isfinite(stack), np.maximum(stack, 0), np.nan)
    flows = []
    if interp == 'flow':
        for i in range(len(epochs) - 1):
            a, b = [np.arcsinh(np.nan_to_num(s) / 0.05) for s in (stack[i], stack[i + 1])]
            flows.append(optical_flow_tvl1(a, b, attachment=10))
            print(f"flow {epochs['obs'][i]} -> {epochs['obs'][i + 1]}: median "
                  f"{np.median(np.hypot(*flows[-1])) * binning:.1f} pixels")
    # line of sight interpolation between visits, on an isotropic depth grid
    nz = int(np.ceil(np.max(depth) / vox_mpc)) + 1
    vol = np.zeros((nz, ny, nx), 'float32')
    for k in range(nz):
        dk = k * vox_mpc
        for i in range(len(epochs) - 1):
            inside = (depth[i] <= dk) & (dk <= depth[i + 1])
            if not inside.any():
                continue
            w = np.clip((dk - depth[i]) / (depth[i + 1] - depth[i]), 0, 1)
            if interp == 'flow':
                val = flow_interp(stack[i], stack[i + 1], w, flows[i])
            else:
                if interp == 'nearest':
                    w = np.round(w)
                val = np.nan_to_num((1 - w) * stack[i] + w * stack[i + 1])
            vol[k][inside] = val[inside]
    # world coordinates (mpc): x, y on the sky plane, z toward the observer. the first surface is tilted,
    # approximated by a plane z0 = a x + b y + c, which enters the affine as a shear
    zmpc = (z0 - z0.mean()) * 1000
    A = np.c_[xx.ravel() * vox_mpc, yy.ravel() * vox_mpc, np.ones(xx.size)]
    a, b, c = np.linalg.lstsq(A, zmpc.ravel(), rcond=None)[0]
    affine = np.array([[vox_mpc, 0, 0, 0], [0, vox_mpc, 0, 0], [a * vox_mpc, b * vox_mpc, -vox_mpc, c], [0, 0, 0, 1]])
    img = nib.Nifti1Image(np.transpose(vol, (2, 1, 0)), affine)  # (x, y, depth)
    img.header.set_xyzt_units('mm')  # 1 unit = 1 mpc
    nib.save(img, name + '.nii.gz')
    meta = {'units': 'mpc (NIfTI says mm)', 'voxel_mpc': vox_mpc, 'shape_xyz': [nx, ny, nz], 'region_crop': region,
            'binning': binning, 'distance_pc': distance_pc, 'dust_distance_pc': dust_dist, 'sn_year': sn_year,
            'cas_a': CAS_A.to_string('hmsdms'), 'rho_pc': [float(rho.min()), float(rho.max())],
            'surface_slope': [float(a), float(b)], 'tilt_deg': float(np.degrees(np.arctan(np.hypot(a, b)))),
            'dust_k': dust_k, 'threshold_default': float(np.median([i['threshold_binned'] for i in info])),
            'star_k': star_k, 'star_dilate': star_dilate, 'star_max': star_max, 'sky_percentile': sky_prc,
            'galaxies_removed': static, 'stars': stars,
            'interp': interp, 'visits': info}
    with open(name + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'saved {name}.nii.gz, {nx} x {ny} x {nz} voxels of {vox_mpc:.2f} mpc')
    volume_projections(np.where(vol >= meta['threshold_default'], vol, 0), vox_mpc, epochs, info,
                       name + '_projections.png')
    return name


def volume_projections(vol, vox_mpc, epochs, info, fn, exaggerate=10, slab=25):
    """
    maximum intensity projections: sky view, and two side views through slab voxels at the center (dashed lines),
    depth stretched by exaggerate
    """
    from matplotlib.gridspec import GridSpec
    top = np.max(vol, axis=0)
    # side views: max over a thin slab through the center, a max over the whole width always saturates
    nz, ny, nx = vol.shape
    band = slab // 2
    side_x = np.max(vol[:, ny // 2 - band:ny // 2 + band + 1, :], axis=1)  # depth x x
    side_y = np.max(vol[:, :, nx // 2 - band:nx // 2 + band + 1], axis=2)  # depth x y
    clim = np.percentile(top[top > 0], 99.5) if np.any(top > 0) else 1
    fig = plt.figure(figsize=(12, 9))
    gs = GridSpec(2, 2, width_ratios=[nx, nz * exaggerate], height_ratios=[ny, nz * exaggerate], figure=fig)
    ink = '#333333'
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(top, origin='lower', cmap='magma', vmin=0, vmax=clim,
              extent=[0, nx * vox_mpc, 0, ny * vox_mpc])
    ax.set_title('sky view (max over depth)', fontsize=9, color=ink, loc='left')
    for pos, horizontal in [(ny // 2, True), (nx // 2, False)]:
        line = ax.axhline if horizontal else ax.axvline
        line(pos * vox_mpc, color='#aaaaaa', lw=0.6, ls='--')
    ax.set_ylabel('y (mpc)', fontsize=8, color=ink)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(side_y.T, origin='lower', cmap='magma', vmin=0, vmax=clim, aspect=1 / exaggerate,
              extent=[0, nz * vox_mpc, 0, ny * vox_mpc])
    ax.set_title(f'side, vertical line (depth x{exaggerate})', fontsize=9, color=ink, loc='left')
    ax.set_xlabel('depth (mpc)', fontsize=8, color=ink)
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(side_x, origin='lower', cmap='magma', vmin=0, vmax=clim, aspect=exaggerate,
              extent=[0, nx * vox_mpc, 0, nz * vox_mpc])
    ax.set_title(f'side, horizontal line (depth x{exaggerate})', fontsize=9, color=ink, loc='left')
    ax.set_xlabel('x (mpc)', fontsize=8, color=ink)
    ax.set_ylabel('depth (mpc)', fontsize=8, color=ink)
    for d in [i['depth_mpc_mean'] for i in info]:
        ax.axhline(d, color='#aaaaaa', lw=0.5, ls=':')
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    ax.text(0, 1, '\n'.join(f"{i['date']}  {i['depth_mpc_mean']:5.1f} mpc" for i in info), fontsize=8, color=ink,
            va='top', family='monospace')
    for a in fig.axes:
        a.tick_params(labelsize=7, colors=ink)
    fig.tight_layout()
    fig.savefig(fn, dpi=150)
    plt.close(fig)
    print(f'saved {fn}')


def dust_cmap(name):
    """ 'brown': black - brown - orange - cream, like the F444W dust in the 2D images. 'gray', or any matplotlib name """
    if name == 'brown':
        return matplotlib.colors.LinearSegmentedColormap.from_list(
            'brown', ['#000000', '#2e1404', '#6e300a', '#b85c16', '#e8a050', '#ffe2b8'])
    return matplotlib.colormaps[name]


def add_mouse_controls(plotter, center, deg_per_pixel=0.3):
    """
    mouse controls with a center of rotation, as in ParaView or MeshLab:
    left click + drag: the point under the click becomes the center of rotation (VTK volume picker, first voxel
           above the threshold, or where the ray crosses the mid-depth plane if there is no dust), then the camera
           orbits around it by deg_per_pixel, whatever the zoom. the default trackball orbits the focal point,
           with steps relative to the view that become extreme when zoomed in.
    wheel: zoom toward the point under the cursor (VTK DollyToPosition), which also becomes the center.
    shift + left drag (pan), ctrl + left drag (spin): VTK defaults. r: back to the opening view.
    observers on the interactor style replace its built-in handlers for these events.
    returns a dict with rotate_world(axis, degrees), reset(), show_center(bool)
    """
    import vtk
    import pyvista as pv
    iren = plotter.iren.interactor
    style = iren.GetInteractorStyle()
    ren = plotter.renderer
    ctl = {'point': np.array(center, float), 'center': np.array(center, float), 'last': None,
           'show': False, 'camera': plotter.camera_position}
    picker = vtk.vtkVolumePicker()
    picker.SetTolerance(0.0)
    radius = 0.004 * np.linalg.norm(np.array(center)) + 1e-6
    marker = plotter.add_mesh(pv.Sphere(radius=radius, center=center), color='cyan', pickable=False, reset_camera=False)
    marker.SetVisibility(False)

    def set_point(p):
        ctl['point'] = np.asarray(p, float)
        marker.SetPosition(*(ctl['point'] - ctl['center']))

    def sheet_point(x, y):
        # no dust under the cursor: where the cursor ray crosses the mid-depth plane of the (thin) volume
        ends = []
        for z in (0.0, 1.0):
            ren.SetDisplayPoint(x, y, z)
            ren.DisplayToWorld()
            w = ren.GetWorldPoint()
            ends.append(np.array(w[:3]) / w[3])
        near, far = ends
        if abs(far[2] - near[2]) < 1e-9:
            return ctl['point']
        t = (ctl['center'][2] - near[2]) / (far[2] - near[2])
        return near + t * (far - near)

    def pick(x, y):
        if picker.Pick(x, y, 0, ren) and picker.GetVolume() is not None:
            set_point(picker.GetPickPosition())
        else:
            set_point(sheet_point(x, y))

    def wheel(obj, event):
        x, y = iren.GetEventPosition()
        factor = 1.15 if event == 'MouseWheelForwardEvent' else 1 / 1.15
        style.DollyToPosition(factor, [x, y], ren)
        pick(x, y)
        ren.ResetCameraClippingRange()
        plotter.render()

    def orbit(transform):
        cam = plotter.camera
        cam.SetPosition(transform.TransformPoint(cam.GetPosition()))
        cam.SetFocalPoint(transform.TransformPoint(cam.GetFocalPoint()))
        cam.SetViewUp(transform.TransformVector(cam.GetViewUp()))
        cam.OrthogonalizeViewUp()
        ren.ResetCameraClippingRange()

    def around_point(rotations):
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Translate(*(-ctl['point']))
        for deg, axis in rotations:
            t.RotateWXYZ(deg, *axis)
        t.Translate(*ctl['point'])
        return t

    def rotate(dx, dy):
        cam = plotter.camera
        up = np.array(cam.GetViewUp())
        right = np.cross(np.array(cam.GetDirectionOfProjection()), up)
        orbit(around_point([(-dx * deg_per_pixel, up), (dy * deg_per_pixel, right)]))

    def rotate_world(axis, degrees):
        """ rotate the volume by degrees around a world axis (0 x, 1 y, 2 depth) through the center of rotation """
        vec = np.zeros(3)
        vec[axis] = 1
        orbit(around_point([(-degrees, vec)]))  # the camera turns the other way
        plotter.render()

    def reset():
        plotter.camera_position = ctl['camera']
        set_point(ctl['center'])
        ren.ResetCameraClippingRange()
        plotter.render()

    def show_center(show):
        ctl['show'] = bool(show)
        marker.SetVisibility(ctl['show'])
        plotter.render()

    def press(obj, event):
        if iren.GetShiftKey() or iren.GetControlKey():
            style.OnLeftButtonDown()  # pan / spin
            return
        x, y = iren.GetEventPosition()
        pick(x, y)
        ctl['last'] = (x, y)
        plotter.render()

    def move(obj, event):
        if ctl['last'] is None:
            style.OnMouseMove()
            return
        x, y = iren.GetEventPosition()
        rotate(x - ctl['last'][0], y - ctl['last'][1])
        ctl['last'] = (x, y)
        plotter.render()

    def release(obj, event):
        if ctl['last'] is None:
            style.OnLeftButtonUp()
            return
        ctl['last'] = None

    def key(obj, event):
        if iren.GetKeySym() in ('r', 'R'):
            reset()  # the opening view, instead of the VTK reset that keeps the current orientation
            return
        style.OnChar()

    style.AddObserver('MouseWheelForwardEvent', wheel)
    style.AddObserver('MouseWheelBackwardEvent', wheel)
    style.AddObserver('LeftButtonPressEvent', press)
    style.AddObserver('MouseMoveEvent', move)
    style.AddObserver('LeftButtonReleaseEvent', release)
    style.AddObserver('CharEvent', key)
    ctl.update(rotate_world=rotate_world, reset=reset, show_center=show_center)
    return ctl


def view(name, z_scale=1, clim_prc=99.5, screenshot=None, threshold=None, smooth=0, cmap='brown'):
    """
    rotate the dust volume in 3D, pyvistaqt window (as MNE) with a Controls panel: info, threshold (dust below it is
    transparent), color max (brightness), depth smoothing (gaussian along the line of sight), colormap.
    the panel can be closed and reopened (View menu, Ctrl+T) or dragged out as a separate window.
    mouse: left drag rotates, the wheel zooms toward the cursor (the rotation center follows), shift + drag pans.
    z_scale stretches depth (1 = true proportions).
    """
    import nibabel as nib
    import pyvista as pv
    from pyvistaqt import BackgroundPlotter
    from PyQt5 import QtWidgets, QtCore
    from scipy.ndimage import gaussian_filter1d
    data = np.asarray(nib.load(name + '.nii.gz').dataobj, dtype='float32')  # (x, y, depth), Fortran order
    with open(name + '.json') as f:
        meta = json.load(f)
    vox = meta['voxel_mpc']  # the affine zooms include the shear
    nx, ny, nz = data.shape
    thr0 = meta.get('threshold_default', 0) if threshold is None else threshold
    pos = data[data > thr0]
    top0 = float(np.percentile(pos, clim_prc)) if len(pos) else 1.0
    vlim = float(np.percentile(pos, 99.99)) if len(pos) else 1.0  # largest color max offered
    # clip to the largest color max: a smaller range keeps the GPU lookup tables small
    data = np.clip(data, 0, vlim)
    grid = pv.ImageData(dimensions=data.shape, spacing=(vox, vox, vox * z_scale))
    grid.point_data['dust'] = data.ravel(order='F').copy()  # a copy, not a view: smoothing must not change data
    size = (f'x {nx * vox:.0f} mpc ({nx} voxels)\ny {ny * vox:.0f} mpc ({ny} voxels)\n'
            f'depth {nz * vox:.1f} mpc ({nz} voxels)\nvoxel {vox:.2f} mpc, depth shown x{z_scale:g}')
    print(f'{os.path.basename(name)}: ' + size.replace('\n', ', '))
    plotter = BackgroundPlotter(title=os.path.basename(name), window_size=(1400, 900), off_screen=bool(screenshot),
                                toolbar=False)  # own toolbar below, the default views along the axes are not useful
    plotter.set_background('black')
    actor = plotter.add_volume(grid, scalars='dust', cmap=dust_cmap(cmap), clim=[0, top0],
                               scalar_bar_args={'title': 'MJy/sr above sky', 'color': 'white'})
    state = {'thr': thr0, 'top': top0, 'smooth': smooth, 'cmap': cmap}

    def update_opacity():
        # transparent below the threshold, a short ramp (VTK sizes its lookup table by the closest pair of points),
        # then rising to the color max, and flat above it
        top = state['top']
        ramp = 0.02 * top
        thr = min(max(state['thr'], 0), top - 2 * ramp)
        otf = actor.GetProperty().GetScalarOpacity()
        otf.RemoveAllPoints()
        otf.AddPoint(0, 0)
        otf.AddPoint(thr, 0)
        otf.AddPoint(thr + ramp, 0.05)
        otf.AddPoint(top, 0.8)
        otf.AddPoint(max(vlim, top * 1.01), 0.8)

    def update_colors():
        lut = actor.mapper.lookup_table
        lut.apply_cmap(dust_cmap(state['cmap']), 256)
        lut.scalar_range = (0, state['top'])
        actor.prop.apply_lookup_table(lut)
        update_opacity()

    def update_smooth():
        s = state['smooth']
        sm = gaussian_filter1d(data, s / vox, axis=2) if s > 0 else data
        grid.point_data['dust'][:] = sm.ravel(order='F')
        grid.Modified()

    update_colors()
    if smooth:
        update_smooth()
    # only the depth axis, labelled in true mpc also when depth is stretched
    plotter.show_bounds(show_xaxis=False, show_yaxis=False, ztitle='depth (mpc)', color='white', font_size=10,
                        location='outer', ticks='outside', n_zlabels=2, fmt='%.0f',
                        axes_ranges=[0, (nx - 1) * vox, 0, (ny - 1) * vox, 0, (nz - 1) * vox])
    plotter.show_axes()

    # wheel zooms toward the point under the cursor (VTK DollyToPosition, like zooming a map). the camera focal
    # point, which is the rotation center, moves toward that point with each step
    ctl = None
    if plotter.iren is not None:  # None off screen
        plotter.render()  # the opening view, restored by reset
        ctl = add_mouse_controls(plotter, [(nx - 1) * vox / 2, (ny - 1) * vox / 2, (nz - 1) * vox * z_scale / 2])
        toolbar = plotter.app_window.addToolBar('Rotate')
        for axis, label in enumerate(['x', 'y', 'depth']):
            for deg in (-5, 5):
                act = toolbar.addAction(f'{label} {deg:+d}\u00b0')
                act.triggered.connect(lambda checked=False, a=axis, d=deg: ctl['rotate_world'](a, d))
            toolbar.addSeparator()
        toolbar.addAction('Reset view').triggered.connect(lambda checked=False: ctl['reset']())

    # Controls panel
    panel = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(panel)
    info = QtWidgets.QLabel(f'<b>{os.path.basename(name)}</b><br>' + size.replace('\n', '<br>') +
                            f"<br>dust threshold default {meta.get('threshold_default', 0):.4f} MJy/sr")
    info.setWordWrap(True)
    layout.addWidget(info)

    def add_slider(title, vmin, vmax, value, key, fmt, on_change, power=1.0):
        """ slider of 1000 steps; power > 1 gives finer steps at the low end """
        label = QtWidgets.QLabel()
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, 1000)
        to_value = lambda i: vmin + (vmax - vmin) * (i / 1000) ** power
        to_index = lambda v: int(round(1000 * ((min(max(v, vmin), vmax) - vmin) / (vmax - vmin)) ** (1 / power)))

        def changed(i):
            state[key] = to_value(i)
            label.setText(f'{title}: {fmt % state[key]}')
            on_change()
            plotter.render()

        slider.setValue(to_index(value))
        label.setText(f'{title}: {fmt % value}')
        slider.valueChanged.connect(changed)
        layout.addWidget(label)
        layout.addWidget(slider)

    add_slider('threshold (MJy/sr)', 0, vlim, thr0, 'thr', '%.4f', update_opacity, power=3)
    add_slider('color max (MJy/sr)', 0.04 * vlim, vlim, top0, 'top', '%.3f', update_colors, power=2)
    add_slider('depth smoothing (mpc)', 0, 10, smooth, 'smooth', '%.1f', update_smooth)
    layout.addWidget(QtWidgets.QLabel('colormap'))
    combo = QtWidgets.QComboBox()
    combo.addItems(['brown', 'gray', 'afmhot', 'magma'])
    combo.setCurrentText(cmap)

    def cmap_changed(text):
        state['cmap'] = text
        update_colors()
        plotter.render()

    combo.currentTextChanged.connect(cmap_changed)
    layout.addWidget(combo)
    if ctl is not None:
        check = QtWidgets.QCheckBox('show center of rotation')
        check.setChecked(False)
        check.toggled.connect(ctl['show_center'])
        layout.addWidget(check)
    layout.addWidget(QtWidgets.QLabel('<br><b>mouse</b><br>left drag: rotate around the clicked point<br>wheel: zoom toward the cursor<br>shift + drag: pan'
                                      '<br>r or Reset view: the opening view<br>toolbar: rotate 5\u00b0 around an axis<br><br>Ctrl+T: show / hide this panel'))
    layout.addStretch()
    dock = QtWidgets.QDockWidget('Controls', plotter.app_window)
    dock.setWidget(panel)
    dock.setMinimumWidth(260)
    plotter.app_window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
    toggle = dock.toggleViewAction()
    toggle.setShortcut('Ctrl+T')
    menus = {a.text(): a.menu() for a in plotter.main_menu.actions() if a.menu() is not None}
    (menus.get('View') or plotter.main_menu.addMenu('View')).addAction(toggle)
    if screenshot:
        plotter.camera.azimuth = 30
        plotter.camera.elevation = 25
        plotter.render()
        plotter.screenshot(screenshot)
        panel.grab().save(screenshot.replace('.png', '_panel.png'))
        print(f'saved {screenshot}')
        plotter.close()
        return
    plotter.app.exec_()


def usage(parser):
    parser.print_help()
    print()
    try:
        list_targets(get_table())
    except Exception as err:
        print(f'could not list targets: {err}')


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='light_echo.py', description=f'Light echo pipeline for JWST program {PROGRAM}.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('target', nargs='?', help='target name, e.g. Greebo (includes Greebo-Repeated)')
    parser.add_argument('--steps', default='download,images,echo', help=f'comma separated steps out of {",".join(STEPS)}')
    parser.add_argument('--ref', default=None,
                        help='reference visit (obs id, e.g. o017) for alignment and stretch. default: earliest '
                             'visit, or the one stored in params.json')
    parser.add_argument('--lims', nargs=2, type=float, default=[0.03, 0.995],
                        help='quantiles of the reference visit mapped to black and white (first run only)')
    parser.add_argument('--stretch', default='asinh', choices=['asinh', 'linear', 'level_adjust'],
                        help='fixed stretch between the black and white values (first run only)')
    parser.add_argument('--asinh-k', type=float, default=10,
                        help='asinh softening, larger lifts faint values more (first run only)')
    parser.add_argument('--factor', type=float, default=2, help='level_adjust factor (first run only)')
    parser.add_argument('--blue-sky', type=float, default=0.04,
                        help='display level (0-1) of the median sky in the blue filter, sets its black point '
                             '(first run only)')
    parser.add_argument('--reset-params', action='store_true', help='recompute params.json')
    parser.add_argument('--no-register', action='store_true', help='WCS reprojection only, no star registration')
    parser.add_argument('--no-fill', action='store_true', help='do not fill holes (star cores) before reprojection')
    parser.add_argument('--overwrite', action='store_true', help='regenerate existing images')
    parser.add_argument('--redo', action='store_true', help='recompute cached aligned layers, implies --overwrite')
    parser.add_argument('--echo-mode', default='sum', choices=['sum', 'mean', 'max'],
                        help='how to combine f444w visits in the echo step')
    parser.add_argument('--region', nargs=4, type=int, default=None, metavar=('Y1', 'Y2', 'X1', 'X2'),
                        help='volume region in crop pixels. default: right of the gap between the NIRCam modules')
    parser.add_argument('--bin', type=int, default=2, help='volume voxel size in pixels (1 pixel ~ 1 mpc)')
    parser.add_argument('--dust-k', type=float, default=5, help='volume: dust threshold in noise levels')
    parser.add_argument('--star-k', type=float, default=10, help='volume: star mask threshold in blue noise levels')
    parser.add_argument('--distance', type=float, default=3400, help='distance to Cas A, pc (Reed et al. 1995)')
    parser.add_argument('--sn-year', type=float, default=1681.0,
                        help='year the Cas A explosion light reached Earth (1681 +- 19, Rest et al. 2008)')
    parser.add_argument('--stars', default='mask', choices=['mask', 'starnet'],
                        help='volume / view: star removal, our mask and fill, or StarNet2 (saved with _starnet)')
    parser.add_argument('--no-static', action='store_true',
                        help='volume / view: keep background galaxies (raw), saved as a separate _raw volume')
    parser.add_argument('--interp', default='flow', choices=['flow', 'linear', 'nearest'],
                        help='volume: line of sight interpolation between visits, flow = motion compensated')
    parser.add_argument('--z-scale', type=float, default=1, help='view: depth exaggeration (1 = true proportions)')
    parser.add_argument('--cmap', default='brown', help='view: colormap, brown (like the 2D images), gray, afmhot, ...')
    parser.add_argument('--changes', action='store_true',
                        help='echo step also saves _changes.jpg, the per-pixel minimum over visits subtracted')
    args = parser.parse_args(argv)
    if args.target is None:
        usage(parser)
        return
    steps = args.steps.split(',')
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise SystemExit(f'unknown steps {bad}, choose from {STEPS}')
    table = get_table()
    base = resolve_target(args.target, table)
    if 'download' in steps:
        download(base, table)
    if 'images' in steps:
        images(base, ref_obs=args.ref, lims=tuple(args.lims), stretch=args.stretch, asinh_k=args.asinh_k,
               factor=args.factor, blue_sky=args.blue_sky, reset_params=args.reset_params,
               do_register=not args.no_register, redo=args.redo, overwrite=args.overwrite, fill=not args.no_fill)
    if 'sources' in steps:
        sources(base, ref_obs=args.ref, region=args.region, do_register=not args.no_register, fill=not args.no_fill,
                starnet=args.stars == 'starnet')
    name = None
    if 'volume' in steps or 'view' in steps:
        name = volume(base, ref_obs=args.ref, region=args.region, binning=args.bin, dust_k=args.dust_k,
                      star_k=args.star_k, distance_pc=args.distance, sn_year=args.sn_year, interp=args.interp,
                      do_register=not args.no_register, fill=not args.no_fill,
                      overwrite=args.overwrite and 'volume' in steps, static=not args.no_static, stars=args.stars)
    if 'view' in steps:
        view(name, z_scale=args.z_scale, cmap=args.cmap)
    if 'echo' in steps:
        echo(base, ref_obs=args.ref, mode=args.echo_mode, do_register=not args.no_register,
             overwrite=args.overwrite or args.redo, fill=not args.no_fill, changes=args.changes)


if __name__ == '__main__':
    main()
