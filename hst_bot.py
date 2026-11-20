"""HST bot: posts new Hubble images of named objects to Bluesky (https://bsky.app/profile/astrobothst.bsky.social).

HST takes lots of data with little to see, so unlike the JWST bot it posts only images that show a named object:
  - new rows of docs/hst/latest.csv (hst_news_csv.py), grouped by target and pointing
  - SIMBAD objects of named catalogs (M, NGC, IC, Arp, UGC, ESO, ACO = Abell, Mrk, LEDA, PGC) inside the image,
    labelled on the image and named in the post. More than MAX_LABELS: only the target (if it is one of them),
    else LEDA / PGC are dropped.
  - a target name that SIMBAD knows as a named object (e.g. NGC-5905) is posted even when its label is out of frame
  - moving targets (MTFLAG = T, comets, planets) are always posted: images of the last visit, aligned by pixels
  - skipped: UV-only data (all filters < MIN_WAVE nm, nearly empty images) and subarray strips
  - 2 or more filters make a color image (R = reddest, B = bluest, G = middle or the mean of R and B), 1 filter grey
Image processing comes from docs/color_lab.py (range requests to the STScI S3 mirror, cosmic-ray removal,
hole filling, reprojection, level_adjust). Posts are logged in docs/bot_hst_posts.csv; a group is processed again
only when it has new files, and posted again only when it has new filters.

Usage:
    python hst_bot.py               # post new data (GitHub Action, after hst_news_csv.py)
    python hst_bot.py --dry         # make the images and texts in data/tmp/hst_bot, no posting, no log
Secrets (env): BNDL_HST (handle or email), BWORD_HST (app password)
"""
import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'docs'))
import color_lab as cl  # noqa: E402

LATEST = os.path.join(HERE, 'docs', 'hst', 'latest.csv')
LOG = os.path.join(HERE, 'docs', 'bot_hst_posts.csv')
LOG_COLUMNS = ['target_name', 'release', 'files', 'filters', 'labels', 'posted']
OUT = os.path.join(HERE, 'data', 'tmp', 'hst_bot')
# CDS cybersecurity incident (Nov 2026): simbad.cds.unistra.fr 404, Harvard mirror unreachable; IDIA (South Africa) works
SIMBAD_HOSTS = ['simbad.idia.ac.za', 'simbad.cfa.harvard.edu', 'simbad.harvard.edu', 'simbad.cds.unistra.fr']
# named catalogs, in order of preference for an object's label: (LIKE prefix for SIMBAD, full-match regex)
CATALOGS = [('M %', r'M \d+'), ('NGC %', r'NGC \d+[A-Z]?'), ('IC %', r'IC \d+[A-Z]?'), ('Arp %', r'Arp \d+'),
            ('UGC %', r'UGC \d+[A-Z]?'), ('ESO %', r'ESO \d+-\d+[A-Z]?'), ('ACO %', r'ACO (S )?\d+'),
            ('Mrk %', r'Mrk \d+[A-Z]?'), ('LEDA %', r'LEDA \d+'), ('PGC %', r'PGC \d+')]
FAINT = ('LEDA', 'PGC')  # dropped first when a field has too many labels
MAX_LABELS = 5
MAX_MP = 4  # megapixels of the reference grid (block-averaged on download)
MAX_FILES_PER_FILTER = 4
MAX_MOVING_FILES = 9  # median of up to this many frames of a moving target (removes cosmic rays)
BSKY_LIMIT = 300
NEWS_URL = 'https://yuval-harpaz.github.io/astro/hst_latest_release.html'
NEWS_TEXT = 'Take a look at the news page.'  # "news page" links to NEWS_URL
CREDIT = 'Credits: NASA, ESA, STScI.'
REPOST_DAYS = 7
MIN_WAVE = 350  # nm: UV-only data (f225w, f275w, f336w) shows almost nothing
MIN_ASPECT = 0.35  # subarray strips


# ---------- names ----------
def clean_id(sid):
    """SIMBAD 'NGC  5905' -> 'NGC 5905'"""
    return ' '.join(str(sid).split())


def norm(name):
    """For comparing names: NGC-5905, NGC 5905 and ngc5905 are the same, NGC0034 = NGC 34; Abell = ACO."""
    s = re.sub(r'[^a-z0-9]', '', str(name).lower()).replace('abell', 'aco')
    return re.sub(r'(?<=[a-z])0+(?=\d)', '', s)


def catalog_rank(sid):
    """Index in CATALOGS of a named-catalog identifier, None for other identifiers."""
    sid = clean_id(sid)
    for k, (_, rx) in enumerate(CATALOGS):
        if re.fullmatch(rx, sid):
            return k
    return None


# ---------- SIMBAD (TAP) ----------
def simbad(adql):
    err = None
    for host in SIMBAD_HOSTS:
        try:
            r = requests.get(f'https://{host}/simbad/sim-tap/sync', timeout=(15, 180),
                             params={'request': 'doQuery', 'lang': 'adql', 'format': 'json', 'query': adql})
            r.raise_for_status()
            j = r.json()
            return pd.DataFrame(j['data'], columns=[c['name'] for c in j['metadata']])
        except Exception as e:
            err = e
            print(f'SIMBAD {host} failed: {e}')
    raise RuntimeError(f'SIMBAD unavailable: {err}')


def _best(rows):
    """rows: one SIMBAD object, one row per identifier -> the best named-catalog label or None."""
    ranked = [(catalog_rank(i), clean_id(i)) for i in rows['id']]
    ranked = [r for r in ranked if r[0] is not None]
    return min(ranked)[1] if ranked else None


def named_objects(ra, dec, rad):
    """Objects with a named-catalog identifier within rad degrees: DataFrame oid, label, rank, ra, dec, otype, ids (normalized)."""
    like = ' OR '.join(f"i.id LIKE '{p}'" for p, _ in CATALOGS)
    df = simbad(f"SELECT TOP 5000 b.oid, b.ra, b.dec, b.otype, i.id FROM basic AS b JOIN ident AS i ON i.oidref = b.oid "
                f"WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', {ra}, {dec}, {rad})) = 1 AND ({like})")
    out = []
    for oid, rows in df.groupby('oid'):
        lab = _best(rows)
        if lab:
            out.append({'oid': int(oid), 'label': lab, 'rank': catalog_rank(lab), 'ra': float(rows['ra'].iloc[0]),
                        'dec': float(rows['dec'].iloc[0]), 'otype': rows['otype'].iloc[0],
                        'ids': {norm(i) for i in rows['id']}})
    return pd.DataFrame(out, columns=['oid', 'label', 'rank', 'ra', 'dec', 'otype', 'ids'])


def resolve_target(target):
    """The SIMBAD object named like the HST target name, if it has a named-catalog identifier:
    dict oid, label, ra, dec, or None. NGC-5905 -> NGC 5905, ABELL-S1063 -> ACO S 1063."""
    cands = list(dict.fromkeys([target.replace('-', ' '), target, target.replace('-', '')]))
    cands = [c.replace("'", "''") for c in cands if c.strip()]
    where = ' OR '.join(f"i.id = '{c}'" for c in cands)
    try:
        df = simbad(f"SELECT TOP 1 b.oid, b.ra, b.dec FROM ident AS i JOIN basic AS b ON i.oidref = b.oid WHERE {where}")
        if not len(df):
            return None
        oid = int(df['oid'].iloc[0])
        ids = simbad(f'SELECT id FROM ident WHERE oidref = {oid}')
    except Exception as e:
        print(f'resolve {target}: {e}')
        return None
    same = [clean_id(i) for i in ids['id'] if catalog_rank(i) is not None and norm(i) == norm(target)]
    lab = same[0] if same else _best(ids)  # NGC0034 -> NGC 34, not its other name NGC 17
    if not lab:
        return None
    return {'oid': oid, 'label': lab, 'ra': float(df['ra'].iloc[0]), 'dec': float(df['dec'].iloc[0])}


# ---------- data ----------
def wave_nm(filt, instrument):
    """F606W -> 606 nm; WFC3/IR filter numbers are in units of 10 nm (F160W = 1.6 um)."""
    m = re.search(r'\d+', filt)
    n = int(m.group()) if m else 0
    return n * 10 if str(instrument).startswith('WFC3/IR') else n


def load_rows(days=7, csv=LATEST):
    """Rows of hst/latest.csv released in the last `days` days, with file, filt (lowercase), wave (nm)."""
    df = pd.read_csv(csv, dtype={'obsid': str, 'proposal_id': str})
    df = df[pd.to_datetime(df['t_obs_release']) > pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=days)].copy()
    df['file'] = df['dataURL'].str.split('/').str[-1]
    df['filt'] = df['filters'].fillna('').str.extract(r'(F\d{3,4}[A-Z]*)', expand=False).fillna('').str.lower()
    df = df[df['filt'] != ''].copy()
    df['wave'] = [wave_nm(f, i) for f, i in zip(df['filt'], df['instrument_name'])]
    return df.reset_index(drop=True)


def _sep(ra1, dec1, ra2, dec2):
    r1, d1, r2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    return np.degrees(np.arccos(np.clip(np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2), -1, 1)))


def is_moving_file(fname):
    try:
        _, hdr, _, _ = cl._read_header(cl.s3url(fname), 0)
        return cl.is_moving(hdr)
    except Exception as e:
        print(f'{fname}: could not read primary header ({e})')
        return False


def make_groups(df):
    """List of DataFrames: per target, rows of the same visit or overlapping pointings (single linkage, center
    distance < half the larger radius). A moving target is one group with the frames of its last visit."""
    groups = []
    for target, rows in df.groupby('target_name', sort=False):
        rows = rows.reset_index(drop=True)
        visit = rows['obsid'].str[:6]  # instrument, program, visit: a combined product and its single exposures
        if is_moving_file(rows['file'].iloc[0]):
            last = rows[visit == visit[pd.to_datetime(rows['t_max']).idxmax()]].copy()
            last['moving'] = True
            groups.append(last.reset_index(drop=True))
            continue
        n = len(rows)
        lab = -np.ones(n, int)
        for i in range(n):
            if lab[i] >= 0:
                continue
            lab[i] = lab.max() + 1
            stack = [i]
            while stack:
                a = stack.pop()
                for b in np.where(lab < 0)[0]:
                    if visit[a] == visit[b] or _sep(rows['s_ra'][a], rows['s_dec'][a], rows['s_ra'][b], rows['s_dec'][b]) < \
                            0.5 * max(rows['s_rad'][a], rows['s_rad'][b]):
                        lab[b] = lab[i]
                        stack.append(b)
        for g in range(lab.max() + 1):
            grp = rows[lab == g].copy()
            grp['moving'] = False
            groups.append(grp.reset_index(drop=True))
    return groups


def choose_rgb(group):
    """-> red, green, blue file lists and the filters used (long to short). Per filter, combined (association)
    products are preferred over single exposures (obsid ending in a letter)."""
    by = {}
    for _, r in group.sort_values('wave', ascending=False).iterrows():
        by.setdefault((r['filt'], r['wave']), []).append(r['file'])
    for k, files in by.items():
        assoc = [f for f in files if f.split('_')[0][-1].isdigit()]
        files = assoc or files
        lim = MAX_MOVING_FILES if group['moving'].iloc[0] else MAX_FILES_PER_FILTER
        if len(files) > lim:  # evenly spaced
            files = [files[int(i)] for i in np.linspace(0, len(files) - 1, lim)]
        by[k] = files
    keys = list(by)  # long to short
    if len(keys) == 1:
        f = by[keys[0]]
        return f, f, f, [keys[0][0]]
    red, blue = keys[0], keys[-1]
    if len(keys) == 2:
        return by[red], [], by[blue], [red[0], blue[0]]
    mid = (red[1] + blue[1]) / 2
    green = min(keys[1:-1], key=lambda k: abs(k[1] - mid))
    return by[red], by[green], by[blue], [red[0], green[0], blue[0]]


def make_image(red, green, blue, group):
    """Download, process and compose. -> uint8 RGB (FITS orientation, y up), reference file name."""
    cl.clear()
    cl.reset_clock()
    filters = {r['file']: [r['filt'], float(r['wave'])] for _, r in group.iterrows()}
    cl.set_filters(filters)
    names = list(dict.fromkeys(red + green + blue))
    for name in names:
        data, text, f = cl.read_sci(name, MAX_MP)
        if not group['file'].str.startswith('jw').any():
            data[data == 0] = np.nan  # HST drizzled products: 0 = no data
        cl._add(name, data, text, f, True, cosmic=True)
    rgb = cl.compose_rgb(red, green, blue, method='rgb', factor=2.0, grey=True, mosaic=True)
    return rgb, red[0]


# ---------- labels ----------
def frame_labels(ref, objects):
    """Objects inside the image data of the reference layer, with pixel x, y (FITS orientation)."""
    if not len(objects):
        return objects.assign(x=[], y=[])
    x, y = cl.sky_to_pix(ref, objects['ra'].values, objects['dec'].values)
    objs = objects.assign(x=x, y=y)
    return objs[np.isfinite(objs['x'])].reset_index(drop=True)


def select_labels(objs, target_obj):
    """More than MAX_LABELS: keep the target if it is in frame, else drop LEDA / PGC, then the best MAX_LABELS."""
    if len(objs) <= MAX_LABELS:
        return objs
    if target_obj is not None and (objs['oid'] == target_obj['oid']).any():
        return objs[objs['oid'] == target_obj['oid']].reset_index(drop=True)
    objs = objs[~objs['label'].str.startswith(FAINT)]
    return objs.sort_values('rank', kind='stable').head(MAX_LABELS).reset_index(drop=True)


def draw_labels(img, objs):
    """img: uint8 RGB, top row first. Labels start right of each object (moved left near the right edge)."""
    import cv2
    img = np.ascontiguousarray(img)
    ny, nx = img.shape[:2]
    scale = max(0.5, max(nx, ny) / 2000 * 0.9)
    th = max(1, int(round(scale * 2)))
    for _, o in objs.iterrows():
        (w, h), _ = cv2.getTextSize(o['label'], cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        x = int(round(o['x'])) + int(6 * scale)
        x = min(max(x, 2), nx - w - 2)
        yy = ny - 1 - int(round(o['y'])) + h // 2
        yy = min(max(yy, h + 2), ny - 3)
        cv2.circle(img, (int(round(o['x'])), ny - 1 - int(round(o['y']))), max(2, th), (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (int(round(o['x'])), ny - 1 - int(round(o['y']))), max(1, th - 1), (100, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, o['label'], (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 3, cv2.LINE_AA)
        cv2.putText(img, o['label'], (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, (100, 255, 255), th, cv2.LINE_AA)
    return img


def save_jpg(img, path, max_bytes=950_000):
    """Bluesky takes images up to 1 MB: lower the quality, then the size, until it fits."""
    from PIL import Image
    im = Image.fromarray(img)
    while True:
        for q in (92, 85, 78, 70):
            im.save(path, quality=q)
            if os.path.getsize(path) <= max_bytes:
                return im.size
        im = im.resize((int(im.width * 0.8), int(im.height * 0.8)), Image.LANCZOS)


# ---------- post text ----------
def in_name(label, target):
    """True when a SIMBAD label is already in the target name (NGC 5905 in NGC-5905, ACO S 1063 in ABELL-S1063)."""
    return re.search(re.escape(norm(label)) + r'(?![0-9])', norm(target)) is not None


def post_texts(target, names, filters, group, crval):
    """-> (main text, reply or None). The target name, then the labels it does not include; ends with a link to the
    news page. Credits go to a reply when the post would be too long, then extra labels are dropped."""
    r = group.iloc[0]
    kind = 'RGB filters' if len(filters) > 1 else 'Filter'
    extra = [n for n in names if not in_name(n, target)]

    def build(extra, more=False, credit=True):
        with_ = f" with {', '.join(extra)}{' and more' if more else ''}" if extra else ''
        return f"\U0001F916 image processing for #Hubble \U0001F52D data ({target}){with_}. {kind}: {', '.join(filters)}. " \
               f"PI: {r['proposal_pi']}, program {r['proposal_id']}. CRVAL: {crval[0]:.6f}, {crval[1]:.6f}." + \
               (f"\n{CREDIT}" if credit else '') + f"\n{NEWS_TEXT}"

    main = build(extra)
    if len(main) <= BSKY_LIMIT:
        return main, None
    main, k = build(extra, credit=False), len(extra)
    while len(main) > BSKY_LIMIT and k > 0:
        k -= 1
        main = build(extra[:k], more=True, credit=False)
    return main, CREDIT


# ---------- Bluesky ----------
_client = None


def bsky():
    global _client
    if _client is None:
        from atproto import Client
        _client = Client()
        _client.login(os.environ['BNDL_HST'], os.environ['BWORD_HST'])
    return _client


def _builder(text, links=None):
    """Text with clickable #hashtags and links: {phrase: url}, default "news page" -> NEWS_URL."""
    from atproto import client_utils
    links = {'news page': NEWS_URL} if links is None else links
    tb = client_utils.TextBuilder()
    pos = 0
    rx = '|'.join([r'#\w+'] + [re.escape(k) for k in links])
    for m in re.finditer(rx, text):
        tb.text(text[pos:m.start()])
        if m.group().startswith('#'):
            tb.tag(m.group(), m.group()[1:])
        else:
            tb.link(m.group(), links[m.group()])
        pos = m.end()
    tb.text(text[pos:])
    return tb


def post(text, image_path, alt, reply=None, size=None):
    """Post an image (and a reply). -> url of the post."""
    from atproto import models
    c = bsky()
    with open(image_path, 'rb') as f:
        data = f.read()
    ar = models.AppBskyEmbedDefs.AspectRatio(width=size[0], height=size[1]) if size else None
    p = c.send_image(text=_builder(text), image=data, image_alt=alt, image_aspect_ratio=ar)
    if reply:
        ref = models.create_strong_ref(p)
        c.send_post(text=_builder(reply), reply_to=models.AppBskyFeedPost.ReplyRef(parent=ref, root=ref))
    return f"https://bsky.app/profile/{c.me.handle}/post/{p.uri.split('/')[-1]}"


# ---------- one group ----------
def read_log():
    if os.path.isfile(LOG):
        return pd.read_csv(LOG, dtype=str).fillna('')
    return pd.DataFrame(columns=LOG_COLUMNS)


def is_new(group, log):
    done = set(';'.join(log['files'][~log['posted'].str.startswith('failed')]).split(';'))  # failed: try again
    return not set(group['file']).issubset(done)


def already_posted(target, filters, log):
    """True when the target was posted in the last REPOST_DAYS days with all these filters."""
    rows = log[(log['target_name'] == target) & log['posted'].str.startswith('http')]
    rows = rows[pd.to_datetime(rows['release']) > pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=REPOST_DAYS)]
    posted = set(';'.join(rows['filters']).split(';'))
    return len(rows) > 0 and set(filters).issubset(posted)


def process(group, do_post=False, log=None, out_dir=OUT):
    """Decide, make the image and (optionally) post one group. -> log row (dict)."""
    target = group['target_name'].iloc[0]
    moving = bool(group['moving'].iloc[0])
    row = {'target_name': target, 'release': group['t_obs_release'].max(), 'files': ';'.join(group['file']),
           'filters': '', 'labels': '', 'posted': ''}
    red, green, blue, filters = choose_rgb(group)
    row['filters'] = ';'.join(filters)
    tag = f"{target} ({', '.join(filters)})"
    if log is not None and already_posted(target, filters, log):
        row['posted'] = 'skip: already posted'
        print(f'{tag}: already posted')
        return row
    if not moving and group[group['filt'].isin(filters)]['wave'].max() < MIN_WAVE:
        row['posted'] = 'skip: UV only'
        print(f'{tag}: UV only')
        return row
    target_obj = None if moving else resolve_target(target)
    if not moving:
        # quick check on the csv footprint before downloading
        ra, dec = group['s_ra'].mean(), group['s_dec'].mean()
        rad = max(_sep(ra, dec, group['s_ra'], group['s_dec']).max() + group['s_rad'].max(), 0.01)
        objects = named_objects(ra, dec, rad)
        if not len(objects) and target_obj is None:
            row['posted'] = 'skip: no named objects'
            print(f'{tag}: no named objects')
            return row
    print(f'{tag}: making image' + (' (moving target)' if moving else ''))
    rgb, ref = make_image(red, green, blue, group)
    if min(rgb.shape[:2]) < MIN_ASPECT * max(rgb.shape[:2]):
        row['posted'] = 'skip: narrow strip'
        print(f'{tag}: narrow strip (subarray)')
        return row
    labels = pd.DataFrame(columns=['oid', 'label', 'rank', 'x', 'y'])
    if not moving:
        ra, dec, rad = cl.footprint(ref)
        labels = select_labels(frame_labels(ref, named_objects(ra, dec, rad)), target_obj)
        if target_obj is not None:  # the target's own name (NGC 34), not another id of it (NGC 17)
            labels.loc[labels['oid'] == target_obj['oid'], 'label'] = target_obj['label']
        if not len(labels) and target_obj is None:
            row['posted'] = 'skip: no named objects in frame'
            print(f'{tag}: no named objects in frame')
            return row
    names = list(labels['label'])
    if target_obj is not None and target_obj['label'] not in names:
        names = [target_obj['label']] + names
    elif target_obj is not None:
        names = [target_obj['label']] + [n for n in names if n != target_obj['label']]
    row['labels'] = ';'.join(names)
    img = draw_labels(rgb[::-1], labels) if len(labels) else rgb[::-1]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, re.sub(r'[^\w.+-]', '_', f'{target}_{"_".join(filters)}.jpg'))
    size = save_jpg(img, path)
    _, c1, c2, _ = cl.ref_info(ref)
    text, reply = post_texts(target, names, filters, group, (c1, c2))
    alt = f"Hubble image of {', '.join([target] + [n for n in names if not in_name(n, target)])} ({'color, filters' if len(filters) > 1 else 'filter'} " \
          f"{', '.join(filters)}){', labels from SIMBAD' if len(labels) else ''}. Automatic processing."
    print(f'  {path}\n  {text}' + (f'\n  reply: {reply}' if reply else ''))
    if do_post:
        try:
            row['posted'] = post(text, path, alt, reply, size)
            print(f'  posted {row["posted"]}')
        except Exception as e:
            row['posted'] = f'failed: {e}'[:200]
            print(f'  {row["posted"]}')
    else:
        row['posted'] = 'dry'
    return row


def run(days=7, do_post=True, only=None, write_log=True):
    """Process every group with new files. only: list of target names. -> DataFrame of log rows."""
    log = read_log()
    groups = [g for g in make_groups(load_rows(days)) if is_new(g, log)]
    if only:
        groups = [g for g in groups if g['target_name'].iloc[0] in only]
    print(f'{len(groups)} new groups')
    rows = []
    for g in groups:
        try:
            row = process(g, do_post=do_post, log=log)
        except Exception as e:
            print(f"{g['target_name'].iloc[0]}: failed ({e})")
            row = {'target_name': g['target_name'].iloc[0], 'release': g['t_obs_release'].max(),
                   'files': ';'.join(g['file']), 'filters': '', 'labels': '', 'posted': f'failed: {e}'[:200]}
        rows.append(row)
        if write_log and row['posted'] != 'dry':
            log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
            log.to_csv(LOG, index=False)
        if row['posted'].startswith('http'):
            time.sleep(5)
    return pd.DataFrame(rows, columns=LOG_COLUMNS)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--days', type=float, default=3, help='look back this many days of release (default 3)')
    ap.add_argument('--dry', action='store_true', help='make images and texts only (data/tmp/hst_bot)')
    a = ap.parse_args()
    run(a.days, do_post=not a.dry, write_log=not a.dry)
