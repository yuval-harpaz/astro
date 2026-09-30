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
from scipy.ndimage import gaussian_filter, binary_dilation, shift as nd_shift
from skimage.registration import phase_cross_correlation
import warnings
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', FITSFixedWarning)

PROGRAM = '8527'
STEPS = ['download', 'images', 'echo', 'volume', 'view']
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


def star_mask(blue, bvalid, red, rvalid, star_k=10, dust_k=5, star_dilate=2, star_max=60):
    """
    stars are found in the blue filter, where dust is faint, as point sources: above a smoothed version by star_k
    noise levels, so extended dust (visible in F200W) is not masked. each source is grown to cover its F444W halo,
    by star_dilate * (peak / threshold) ** (1 / 3) pixels up to star_max, where peak is the source's compact F444W
    brightness and threshold is dust_k F444W noise levels (PSF wings fall as r^-3). sizing by F444W, not blue,
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
    radii = np.minimum(star_dilate * np.maximum(peaks / thr, 1) ** (1 / 3), star_max)
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


def volume(base, ref_obs=None, region=None, binning=2, dust_k=5, star_k=10, star_dilate=2, sky_prc=10,
           distance_pc=3400, sn_year=1681.0, interp='linear', do_register=True, fill=True, overwrite=False):
    """
    dust density proxy on a voxel grid. x, y: sky pixels (binned), depth: behind the first visit's echo surface.
    per visit: stars masked with the blue filter, sky (percentile sky_prc) subtracted, f444w binned, values below
    dust_k noise levels set to 0. each visit is placed at its echo depth, computed per pixel from the echo
    paraboloid, and the line of sight is interpolated between visits. voxels are isotropic, in mpc.
    saves NIfTI (+ json) and a projections png to out/volume/.
    """
    import nibabel as nib
    epochs, out, params, ref = setup(base, ref_obs)
    if len(epochs) < 2:
        raise SystemExit(f'volume needs repeated visits, {base} has {len(epochs)}')
    if 'crop' not in params:
        raise SystemExit('run the images step first')
    os.makedirs(out + 'volume', exist_ok=True)
    name = f"{out}volume/{base}_dust_bin{binning}_to{epochs['date'].iloc[-1]}"
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
    pix_arcsec = np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600
    vox_mpc = pix_arcsec / 206265 * distance_pc * 1000 * binning
    # binned pixel centers in full reference grid coordinates -> distance from Cas A on the sky (pc)
    ny, nx = (ry2 - ry1) // binning, (rx2 - rx1) // binning
    yy, xx = np.mgrid[0:ny, 0:nx]
    xs = cx1 + rx1 + (xx + 0.5) * binning - 0.5
    ys = cy1 + ry1 + (yy + 0.5) * binning - 0.5
    rho = wcs.pixel_to_world(xs, ys).separation(CAS_A).rad * distance_pc
    stack = np.zeros((len(epochs), ny, nx), 'float32')
    depth = np.zeros((len(epochs), ny, nx))  # mpc behind the first visit's surface
    info = []
    age0 = Time(epochs['mjd'].iloc[0], format='mjd').decimalyear - sn_year
    z0 = echo_depth(rho, age0)
    for iep, ep in epochs.iterrows():
        lay = aligned_layers(ep, ref, out, do_register=do_register, fill=fill)
        red = lay['red'][cy1:cy2, cx1:cx2][ry1:ry2, rx1:rx2].astype(float)
        blue = lay['blue'][cy1:cy2, cx1:cx2][ry1:ry2, rx1:rx2].astype(float)
        valid = np.isfinite(red) & (red != 0)
        sky = np.percentile(red[valid], sky_prc)
        noise = pixel_noise(np.where(valid, red, np.nan))
        bvalid = np.isfinite(blue) & (blue != 0)
        stars = star_mask(blue, bvalid, red, valid, star_k, dust_k, star_dilate)
        dust = np.where(valid & ~stars, red - sky, np.nan)
        dust = block_mean(dust, binning)[:ny, :nx]
        thr = dust_k * noise / binning
        dust[dust < thr] = 0
        stack[iep] = dust
        age = Time(ep['mjd'], format='mjd').decimalyear - sn_year
        depth[iep] = (z0 - echo_depth(rho, age)) * 1000
        info.append({'obs': ep['obs'], 'date': ep['date'], 'age_yr': age, 'sky': sky, 'noise': noise,
                     'threshold_binned': thr, 'star_fraction': float(stars[valid].mean()),
                     'depth_mpc_mean': float(np.mean(depth[iep]))})
        print(f"{ep['obs']} {ep['date']} depth {np.mean(depth[iep]):6.2f} mpc, sky {sky:.3f}, noise {noise:.4f}, "
              f"stars {100 * stars[valid].mean():.1f}%, dust voxels {100 * np.mean(dust > 0):.1f}%")
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
            if interp == 'nearest':
                w = np.round(w)
            val = (1 - w) * stack[i] + w * stack[i + 1]
            vol[k][inside] = np.nan_to_num(val[inside])
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
            'binning': binning, 'distance_pc': distance_pc, 'sn_year': sn_year, 'cas_a': CAS_A.to_string('hmsdms'),
            'rho_pc': [float(rho.min()), float(rho.max())], 'surface_slope': [float(a), float(b)],
            'tilt_deg': float(np.degrees(np.arctan(np.hypot(a, b)))), 'dust_k': dust_k, 'star_k': star_k,
            'star_dilate': star_dilate, 'sky_percentile': sky_prc, 'interp': interp, 'visits': info}
    with open(name + '.json', 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'saved {name}.nii.gz, {nx} x {ny} x {nz} voxels of {vox_mpc:.2f} mpc')
    volume_projections(vol, vox_mpc, epochs, info, name + '_projections.png')
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


def view(name, z_scale=10, clim_prc=99.5, screenshot=None):
    """ rotate the dust volume in 3D with pyvista (the MNE 3D backend). depth stretched by z_scale """
    import nibabel as nib
    import pyvista as pv
    data = np.asarray(nib.load(name + '.nii.gz').dataobj, dtype='float32')  # (x, y, depth)
    with open(name + '.json') as f:
        vox = json.load(f)['voxel_mpc']  # the affine zooms include the shear
    grid = pv.ImageData(dimensions=data.shape, spacing=(vox, vox, vox * z_scale))
    grid.point_data['dust'] = data.ravel(order='F')
    pos = data[data > 0]
    clim = [0, float(np.percentile(pos, clim_prc))] if len(pos) else [0, 1]
    plotter = pv.Plotter(off_screen=screenshot is not None)
    plotter.set_background('black')
    plotter.add_volume(grid, scalars='dust', cmap='magma', clim=clim, opacity='sigmoid_6',
                       scalar_bar_args={'title': 'MJy/sr above sky', 'color': 'white'})
    plotter.add_text(f'{os.path.basename(name)}\ndepth x{z_scale}, voxel {vox:.2f} mpc', font_size=9, color='white')
    plotter.show_axes()
    if screenshot:
        plotter.camera.azimuth = 30
        plotter.camera.elevation = 25
        plotter.screenshot(screenshot)
        print(f'saved {screenshot}')
    else:
        plotter.show()


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
    parser.add_argument('--interp', default='linear', choices=['linear', 'nearest'],
                        help='volume: line of sight interpolation between visits')
    parser.add_argument('--z-scale', type=float, default=10, help='view: depth exaggeration')
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
    name = None
    if 'volume' in steps or 'view' in steps:
        name = volume(base, ref_obs=args.ref, region=args.region, binning=args.bin, dust_k=args.dust_k,
                      star_k=args.star_k, distance_pc=args.distance, sn_year=args.sn_year, interp=args.interp,
                      do_register=not args.no_register, fill=not args.no_fill,
                      overwrite=args.overwrite and 'volume' in steps)
    if 'view' in steps:
        view(name, z_scale=args.z_scale)
    if 'echo' in steps:
        echo(base, ref_obs=args.ref, mode=args.echo_mode, do_register=not args.no_register,
             overwrite=args.overwrite or args.redo, fill=not args.no_fill, changes=args.changes)


if __name__ == '__main__':
    main()
