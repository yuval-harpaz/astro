'''
Match JWST MIRI images of GC_<number> targets to the NIRCam target name that covers
the same piece of sky. In program 10678 MIRI runs in parallel with NIRCam, so the MIRI
footprint of e.g. target GC_129 sits where the NIRCam footprint of a different target
number is, and the target names do not line up between the two instruments.

The match is geometric: the MIRI footprint (s_region) is sampled on a grid and the
NIRCam target whose footprint contains most of those points wins.

Sets where both instruments already have data are listed first, latest release on top.
A set is ready when both instruments have level 3 mosaics, not only level 2 exposures. Then
the bot announces it on Bluesky with a color image made by gc_color_sets.py. The images are
not pushed, they are saved to the drive or to the gitignored data/tmp. What was posted is
kept in the announced and image_post columns of docs/gc_miri_nircam.csv.

Usage:
    python gc_miri_nircam_match.py            # stop after 5 sets, preview csv
    python gc_miri_nircam_match.py 20         # first 20 sets
    python gc_miri_nircam_match.py all        # full match, writes docs/gc_miri_nircam.csv
    python gc_miri_nircam_match.py all image  # make color images of ready sets, no posting
    python gc_miri_nircam_match.py all post image  # post ready sets with their images
    python gc_miri_nircam_match.py all fresh  # ignore the cached MAST query
@Author: Yuval Harpaz
'''
import os
import re
import sys
import astropy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path
from astroquery.mast import Observations
from astro_utils import resize_to_under_2mb
from gc_color_sets import NotReady, filt_str, query_sets, save_set_image

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
# keep trying to make an image this long after the latest release of a set
IMAGE_DAYS = 3
# sets released before this were announced by hand and get no image post, the ones after
# (GC_60, GC_61) were announced by the bot before their level 3 mosaics existed
FIRST_BOT_SET = '2026-09-19'
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
        # level 2 exposures come first, a set is ready when every filter has a level 3 mosaic
        observed = set(rows['filters'][rows['calib_level'] >= 2])
        mosaics = set(rows['filters'][rows['calib_level'] == 3])
        out[target] = {'polys': polys,
                       'ra': ra.mean(),
                       'dec': dec.mean(),
                       'n_images': int(len(use)),
                       'planned': bool(use['planned'].all()),
                       'level3': bool(mosaics) and observed <= mosaics,
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


def previous_state(csv):
    '''(MIRI target, NIRCam target) -> (announced, image_post) as left by the previous run

    announced is the url of the post that announced the set, "manual" for the sets announced
    by hand before the bot did, "bot" for those the bot announced before urls were kept.
    image_post is the url of the post with the color image of the set.
    '''
    if not os.path.isfile(csv):
        return {}
    prev = pd.read_csv(csv, dtype=str).fillna('')
    if 'announced' not in prev.columns:
        # first run that keeps the state. Sets were announced when both dates showed up
        complete = (prev['miri_obs_date'] != '') & (prev['nircam_obs_date'] != '')
        prev['announced'] = ''
        prev.loc[complete, 'announced'] = np.where(prev['set_release'][complete] < FIRST_BOT_SET,
                                                   'manual', 'bot')
        prev['image_post'] = ''
    pairs = zip(prev['miri_target'], prev['nircam_target'])
    return dict(zip(pairs, zip(prev['announced'], prev['image_post'])))


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


def set_text(row, files, follow_up=False):
    '''Bluesky text for one set with its color image'''
    first = ('#JWST \U0001F52D Galactic Center set in color' if follow_up
             else 'New #JWST \U0001F52D Galactic Center set')
    return (f"\U0001F916 {first},\nMIRI {row['miri_target']} + NIRCam {row['nircam_target']}.\n"
            f'RGB filters: {filt_str(files)}\nCredit: NASA, ESA, CSA, STScI.\nsee ')


def set_alt(row, files):
    return (f"Automatic color preview of JWST images near the Galactic Center. MIRI "
            f"{row['miri_target']} is red, NIRCam {row['nircam_target']} is green and blue. "
            f'Filters, red to blue: {filt_str(files)}')


blient = None


def send(txt, jpg=None, alt=None):
    '''Post the text and the link to the csv on Bluesky, with an image if given. Returns
    the post url'''
    global blient
    from atproto import Client as Blient, client_utils
    if blient is None:
        blient = Blient()
        blient.login(os.environ['Bluehandle'], os.environ['Blueword'])
    boot = client_utils.TextBuilder()
    boot.text(txt)
    boot.link(LINK_TEXT, CSV_URL)
    if jpg is None:
        post = blient.send_post(text=boot)
    else:
        # bluesky takes images up to 1MB, this writes tmprs.jpg, which is gitignored
        resize_to_under_2mb(plt.imread(jpg), max_size_mb=0.9)
        with open('tmprs.jpg', 'rb') as f:
            post = blient.send_image(text=boot, image=f.read(), image_alt=alt)
    return 'https://bsky.app/profile/astrobotjwst.bsky.social/post/' + post.uri.split('/')[-1]


table = query_gc()
miri = footprints(table[table['instrument'] == 'MIRI'])
nircam = footprints(table[table['instrument'] == 'NIRCAM'])
print(f'{len(miri)} MIRI targets, {len(nircam)} NIRCam targets, proposals '
      f'{sorted(set(table["proposal_id"].astype(str)))}')

state = previous_state(OUT_CSV)
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
                 'ready': fp['level3'] and nc.get('level3', False),
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
# what was posted about each set is kept from run to run
pairs = list(zip(result['miri_target'], result['nircam_target']))
result['announced'] = [state.get(pair, ('', ''))[0] for pair in pairs]
result['image_post'] = [state.get(pair, ('', ''))[1] for pair in pairs]
# a set is announced once both instruments have level 3 mosaics, with its color image
due = result['ready'] & (result['announced'] != 'manual')
to_announce = result[due & (result['announced'] == '')]
# sets announced before their mosaics existed get their image in a second post
recent = pd.to_datetime(result['set_release'], errors='coerce') > \
    pd.Timestamp.now(tz='UTC').tz_localize(None) - pd.Timedelta(days=IMAGE_DAYS)
to_image = result[due & (result['announced'] != '') & (result['image_post'] == '') & recent]
if len(to_announce):
    print(f'new sets with level 3 data: {", ".join(to_announce["miri_target"])}')
if len(to_image):
    print(f'announced sets waiting for their image: {", ".join(to_image["miri_target"])}')
if limit == 0 and (image or post) and len(to_announce) + len(to_image):
    table3 = query_sets() if image else None
    n_images = 0
    text_only = []
    for irow, set_row in pd.concat([to_announce, to_image]).iterrows():
        follow_up = result.at[irow, 'announced'] != ''
        jpg = None
        if image and n_images < MAX_IMAGES:
            try:
                jpg, files = save_set_image(set_row, table=table3)
                n_images += 1
            except NotReady as e:
                print(f"{set_row['miri_target']} + {set_row['nircam_target']}: {e}, will retry")
            except Exception as e:
                print(f"failed color image for {set_row['miri_target']}: {e}")
        if not post:
            continue
        if jpg is None:
            # announce without the image, it will come in a second post
            if not follow_up:
                text_only.append(irow)
            continue
        try:
            url = send(set_text(set_row, files, follow_up), jpg, set_alt(set_row, files))
            result.at[irow, 'image_post'] = url
            if not follow_up:
                result.at[irow, 'announced'] = url
            print(f'posted {url}')
        except Exception as e:
            print(f"failed bluesky post for {set_row['miri_target']}: {e}")
    if text_only:
        try:
            url = send(announce_text(result.loc[text_only]))
            result.loc[text_only, 'announced'] = url
            print(f'announced without images {url}')
        except Exception as e:
            print(f'failed bluesky post: {e}')
elif post:
    print('nothing new to post')
result = result.drop(columns=['set_release_mjd', 'complete', 'number'])
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
result.to_csv(out_csv, index=False)
print(f'\nwrote {len(result)} rows to {out_csv}, {len(top)} complete sets, '
      f'{int(result["ready"].sum())} with level 3 data')
no_match = result[result['nircam_target'] == '']
if len(no_match):
    print(f'{len(no_match)} MIRI targets with no NIRCam coverage: {", ".join(no_match["miri_target"])}')
