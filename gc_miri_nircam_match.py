'''
Match JWST MIRI images of GC_<number> targets to the NIRCam target name that covers
the same piece of sky. In program 10678 MIRI runs in parallel with NIRCam, so the MIRI
footprint of e.g. target GC_129 sits where the NIRCam footprint of a different target
number is, and the target names do not line up between the two instruments.

The match is geometric: the MIRI footprint (s_region) is sampled on a grid and the
NIRCam target whose footprint contains most of those points wins.

Sets where both instruments already have data are listed first, latest release on top.
When a set is completed by new data (both instruments have dates now, but did not in the
previous docs/gc_miri_nircam.csv) the bot announces it on Bluesky, and with "image" it also
makes a color image of it (gc_color_sets.py), which is neither posted nor pushed.

Usage:
    python gc_miri_nircam_match.py            # stop after 5 sets, preview csv
    python gc_miri_nircam_match.py 20         # first 20 sets
    python gc_miri_nircam_match.py all        # full match, writes docs/gc_miri_nircam.csv
    python gc_miri_nircam_match.py all post   # full match and announce new sets on Bluesky
    python gc_miri_nircam_match.py all image  # also make a color image of each new set
    python gc_miri_nircam_match.py all fresh  # ignore the cached MAST query
@Author: Yuval Harpaz
'''
import os
import re
import sys
import astropy
import numpy as np
import pandas as pd
from matplotlib.path import Path
from astroquery.mast import Observations

OUT_CSV = 'docs/gc_miri_nircam.csv'
PREVIEW_CSV = 'data/gc_miri_nircam_preview.csv'
CACHE = 'data/gc_mast_query.pkl'
CSV_URL = 'https://github.com/yuval-harpaz/astro/blob/main/docs/gc_miri_nircam.csv'
LINK_TEXT = 'gc_miri_nircam.csv'
# where to break the line with the pair names
LINE_WIDTH = 66
# NIRCam pointings further than this from the MIRI footprint centre are not candidates
MAX_SEP_ARCMIN = 10
# real matches score 0.4 - 1, an edge of a neighbouring tile scores ~0.14
MIN_OVERLAP = 0.3
# grid used to sample the MIRI footprint, points per axis of its bounding box
N_GRID = 80
# pairs to name in one Bluesky post
MAX_ANNOUNCE = 6
# color images to make in one run, each takes about a minute
MAX_IMAGES = 3
# bluesky limit is 300 including the link text, the repo plays safe with 250
blim = 250

args = [a.lower() for a in sys.argv[1:]]
fresh = 'fresh' in args
post = 'post' in args
image = 'image' in args
limit = None
for a in args:
    if a.isdigit():
        limit = int(a)
    elif a in ('all', 'full'):
        limit = 0
if limit is None:
    limit = 5
out_csv = OUT_CSV if limit == 0 else PREVIEW_CSV


def query_gc():
    '''All JWST images with a GC_<number> target name, from MAST or from the local cache'''
    if os.path.isfile(CACHE) and not fresh:
        table = pd.read_pickle(CACHE)
        print(f'read {len(table)} rows from {CACHE}')
    else:
        print('querying MAST for target_name GC_*')
        table = Observations.query_criteria(obs_collection='JWST', dataproduct_type='image',
                                            target_name='GC_*').to_pandas()
        os.makedirs('data', exist_ok=True)
        table.to_pickle(CACHE)
        print(f'{len(table)} rows, cached in {CACHE}')
    # GC-MIRI, GC-RC2-MIRI and the like are other programs, keep only GC_<number>
    table = table[table['target_name'].str.match(r'^GC_\d+$')].copy()
    table['instrument'] = table['instrument_name'].str.split('/').str[0]
    table = table[table['instrument'].isin(['MIRI', 'NIRCAM'])]
    # calib_level -1 is a planned observation, it has a footprint but no date yet
    table['planned'] = table['calib_level'] == -1
    return table.reset_index(drop=True)


def polygons(s_region):
    '''s_region string to a list of (ra, dec) vertex arrays, one per POLYGON'''
    polys = []
    for chunk in re.split('POLYGON', str(s_region).upper())[1:]:
        vals = []
        for tok in chunk.split():
            try:
                vals.append(float(tok))
            except ValueError:  # coordinate system name, e.g. ICRS
                continue
        if len(vals) >= 6:
            vals = np.array(vals[:len(vals) // 2 * 2]).reshape(-1, 2)
            polys.append((vals[:, 0], vals[:, 1]))
    return polys


def footprints(table):
    '''Per target name, the union of its footprints, its centre and its dates.

    Executed footprints (calib_level >= 2) are the real sky coverage and are preferred,
    planned targets fall back on the APT footprint.
    '''
    out = {}
    for target, rows in table.groupby('target_name'):
        use = rows[~rows['planned']] if (~rows['planned']).any() else rows
        polys = []
        for s_region in use['s_region'].dropna().unique():
            polys += polygons(s_region)
        if not polys:
            continue
        ra = np.concatenate([p[0] for p in polys])
        dec = np.concatenate([p[1] for p in polys])
        # "use the latest if there are a few"
        obs = use['t_min'].dropna()
        rel = use['t_obs_release'].dropna()
        out[target] = {'polys': polys,
                       'ra': ra.mean(),
                       'dec': dec.mean(),
                       'n_images': int(len(use)),
                       'planned': bool(use['planned'].all()),
                       'mjd': float(obs.max()) if len(obs) else np.nan,
                       'release_mjd': float(rel.max()) if len(rel) else np.nan}
    return out


def to_xy(ra, dec, ra0, dec0):
    '''Local tangent plane in arcsec around (ra0, dec0), good enough over a few arcmin'''
    x = (np.asarray(ra) - ra0) * np.cos(np.radians(dec0)) * 3600
    y = (np.asarray(dec) - dec0) * 3600
    return x, y


def paths(fp, ra0, dec0):
    return [Path(np.column_stack(to_xy(ra, dec, ra0, dec0))) for ra, dec in fp['polys']]


def sample_points(fp, ra0, dec0):
    '''A grid of points inside the footprint, used to measure overlap'''
    pth = paths(fp, ra0, dec0)
    x = np.concatenate([p.vertices[:, 0] for p in pth])
    y = np.concatenate([p.vertices[:, 1] for p in pth])
    gx, gy = np.meshgrid(np.linspace(x.min(), x.max(), N_GRID),
                         np.linspace(y.min(), y.max(), N_GRID))
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = np.zeros(len(pts), bool)
    for p in pth:
        inside |= p.contains_points(pts)
    return pts[inside]


def covered_fraction(pts, fp, ra0, dec0):
    '''Fraction of the sample points that fall inside the footprint'''
    if not len(pts):
        return 0.0
    inside = np.zeros(len(pts), bool)
    for p in paths(fp, ra0, dec0):
        inside |= p.contains_points(pts)
    return float(inside.mean())


def separation(a, b):
    '''Arcmin between two footprint centres'''
    dx = (a['ra'] - b['ra']) * np.cos(np.radians(a['dec']))
    dy = a['dec'] - b['dec']
    return float(np.hypot(dx, dy) * 60)


def iso(mjd):
    if mjd is None or not np.isfinite(mjd):
        return ''
    return astropy.time.Time(mjd, format='mjd').utc.iso[:19]


def sex(ra, dec):
    c = astropy.coordinates.SkyCoord(ra, dec, unit='deg')
    return c.to_string('hmsdms', sep=':', precision=1)


def complete_sets(csv):
    '''MIRI target names that already had data from both instruments in a previous run'''
    if not os.path.isfile(csv):
        return set()
    prev = pd.read_csv(csv)
    both = prev['miri_obs_date'].notna() & prev['nircam_obs_date'].notna()
    return set(prev['miri_target'][both])


def wrap_pairs(shown):
    '''One pair name per item, broken into lines between pairs and never inside one'''
    lines = ['']
    for ipair, pair in enumerate(shown):
        piece = pair + ('.' if ipair == len(shown) - 1 else ',')
        if not lines[-1]:
            lines[-1] = piece
        elif len(lines[-1]) + len(piece) + 1 > LINE_WIDTH:
            lines.append(piece)
        else:
            lines[-1] += ' ' + piece
    return '\n'.join(lines)


def announce_text(new_rows):
    '''The Bluesky text, shortened until it fits together with the link'''
    pairs = [f"MIRI {r['miri_target']} + NIRCam {r['nircam_target']}"
             for _, r in new_rows.iterrows()]
    for n_show in range(min(MAX_ANNOUNCE, len(pairs)), 0, -1):
        shown = pairs[:n_show]
        if n_show < len(pairs):
            shown.append(f'and {len(pairs) - n_show} more')
        txt = (f'\U0001F916 New #JWST \U0001F52D Galactic Center '
               f'{"set" if len(pairs) == 1 else "sets"},\n'
               f'{wrap_pairs(shown)}\n'
               f'Credit: NASA, ESA, CSA, STScI.\nsee ')
        if len(txt) + len(LINK_TEXT) <= blim:
            break
    return txt


def announce(new_rows):
    '''Tell Bluesky that new data completed one or more MIRI / NIRCam sets'''
    from atproto import Client as Blient, client_utils
    txt = announce_text(new_rows)
    blient = Blient()
    blient.login(os.environ['Bluehandle'], os.environ['Blueword'])
    boot = client_utils.TextBuilder()
    boot.text(txt)
    boot.link(LINK_TEXT, CSV_URL)
    return blient.send_post(text=boot)


table = query_gc()
miri = footprints(table[table['instrument'] == 'MIRI'])
nircam = footprints(table[table['instrument'] == 'NIRCAM'])
print(f'{len(miri)} MIRI targets, {len(nircam)} NIRCam targets, proposals '
      f'{sorted(set(table["proposal_id"].astype(str)))}')

was_complete = complete_sets(OUT_CSV)
order = sorted(miri, key=lambda t: int(t.split('_')[1]))
rows = []
for target in order:
    fp = miri[target]
    pts = sample_points(fp, fp['ra'], fp['dec'])
    scores = []
    for nc_target, nc in nircam.items():
        if separation(fp, nc) > MAX_SEP_ARCMIN:
            continue
        frac = covered_fraction(pts, nc, fp['ra'], fp['dec'])
        if frac > 0:
            scores.append((frac, nc_target))
    scores.sort(reverse=True)
    # a weak best score means the MIRI field falls outside the NIRCam mosaic
    matched = bool(scores) and scores[0][0] >= MIN_OVERLAP
    best = scores[0] if matched else (0.0, '')
    # when nothing was matched the best score is reported as the runner up, for inspection
    others = scores[1:] if matched else scores
    runner_up = others[0] if others else (0.0, '')
    nc = nircam.get(best[1], {})
    rows.append({'miri_target': target,
                 'miri_obs_date': iso(fp['mjd']),
                 'nircam_target': best[1],
                 'nircam_obs_date': iso(nc.get('mjd', np.nan)),
                 'miri_ra': round(fp['ra'], 6),
                 'miri_dec': round(fp['dec'], 6),
                 'miri_coord': sex(fp['ra'], fp['dec']),
                 'set_release': '',
                 'miri_release': iso(fp['release_mjd']),
                 'nircam_release': iso(nc.get('release_mjd', np.nan)),
                 'overlap': round(best[0], 3),
                 'runner_up': runner_up[1],
                 'runner_up_overlap': round(runner_up[0], 3),
                 'miri_status': 'planned' if fp['planned'] else 'observed',
                 'nircam_status': 'planned' if nc.get('planned', True) else 'observed',
                 'n_miri_images': fp['n_images'],
                 'n_nircam_images': nc.get('n_images', 0),
                 'set_release_mjd': np.nanmax([fp['release_mjd'], nc.get('release_mjd', np.nan)])
                 if np.isfinite(fp['release_mjd']) or np.isfinite(nc.get('release_mjd', np.nan)) else np.nan})
    print(f"{target:>7} {rows[-1]['miri_obs_date'] or '(planned)':<19} -> "
          f"{best[1] or '(no match)':>7} {rows[-1]['nircam_obs_date'] or '(planned)':<19} "
          f"overlap {best[0]:.2f}  runner-up {runner_up[1] or '-'} {runner_up[0]:.2f}")
    if limit and len(rows) >= limit:
        print(f'\nstopping after {limit} sets, run with "all" for the full match')
        break

result = pd.DataFrame(rows)
# complete sets on top, latest release first, the rest by target number
result['complete'] = (result['miri_obs_date'] != '') & (result['nircam_obs_date'] != '')
result['set_release'] = [iso(mjd) for mjd in result['set_release_mjd']]
result.loc[~result['complete'], 'set_release'] = ''
result['number'] = [int(t.split('_')[1]) for t in result['miri_target']]
top = result[result['complete']].sort_values('set_release_mjd', ascending=False)
rest = result[~result['complete']].sort_values('number')
result = pd.concat([top, rest]).reset_index(drop=True)
new_rows = result[result['complete'] & ~result['miri_target'].isin(was_complete)]
result = result.drop(columns=['set_release_mjd', 'complete', 'number'])
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
result.to_csv(out_csv, index=False)
print(f'\nwrote {len(result)} rows to {out_csv}, {len(top)} complete sets')
no_match = result[result['nircam_target'] == '']
if len(no_match):
    print(f'{len(no_match)} MIRI targets with no NIRCam coverage: {", ".join(no_match["miri_target"])}')

if len(new_rows):
    print(f'new complete sets: {", ".join(new_rows["miri_target"])}')
    if image and limit == 0:
        # the images are not posted and not pushed, OUT_DIR is the drive or gitignored data/tmp
        from gc_color_sets import OUT_DIR, query_sets, save_set_image
        table3 = query_sets()
        for _, set_row in new_rows.head(MAX_IMAGES).iterrows():
            try:
                save_set_image(set_row, table=table3)
            except Exception as e:
                print(f"failed color image for {set_row['miri_target']}: {e}")
        if len(new_rows) > MAX_IMAGES:
            print(f'{len(new_rows) - MAX_IMAGES} more sets have no image, '
                  f'run gc_color_sets.py for them')
    if post and limit == 0:
        try:
            announce(new_rows)
            print('announced on Bluesky')
        except Exception as e:
            print(f'failed bluesky post: {e}')
elif post:
    print('no new complete set to announce')
