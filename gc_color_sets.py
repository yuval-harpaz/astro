'''
Make a color image for every complete MIRI + NIRCam set of the Galactic Center survey
(program 10678), as listed by gc_miri_nircam_match.py in docs/gc_miri_nircam.csv.

A set is one MIRI F770W image and the two NIRCam images (F480M, F212N) of the target name
that covers the same sky, so the RGB is red = MIRI, green = NIRCam long, blue = NIRCam short.
The images are reprojected onto the NIRCam short wavelength grid, cropped to the MIRI
footprint, so the sharp NIRCam pixels are kept and the frame is the overlapping sky.

Nothing is posted, images are written to /media/yuval/PNY/JWST/images for review.

Usage:
    python gc_color_sets.py             # all complete sets that have no image yet
    python gc_color_sets.py 3           # only the first 3 of them
    python gc_color_sets.py GC_116      # one set, by its MIRI target name
    python gc_color_sets.py GC_116 over # overwrite an image that already exists
@Author: Yuval Harpaz
'''
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astroquery.mast import Observations
from reproject import reproject_interp
from skimage import transform
from astro_utils import *

DRIVE_DIR = '/media/yuval/PNY/JWST/images'
# on github there is no drive, data is not pushed (see .gitignore) so images stay there
OUT_DIR = DRIVE_DIR if os.path.isdir(os.path.dirname(DRIVE_DIR)) else 'data/tmp'
SETS_CSV = 'docs/gc_miri_nircam.csv'
MAST_URL = 'https://mast.stsci.edu/portal/Download/file/JWST/product/'
# the color image is saved with no more pixels than this
MAX_PIX = 4000000
# stretch, as in astro_jwst_news_color.py
FACTOR = 2


def query_sets():
    '''Level 3 GC_<number> images, the ones a color image can be made of'''
    table = Observations.query_criteria(obs_collection='JWST', dataproduct_type='image',
                                        calib_level=3, target_name='GC_*').to_pandas()
    table = table[table['target_name'].str.match(r'^GC_\d+$')].copy()
    table = table[table['dataURL'].notna()]
    table['file'] = [x.replace('mast:JWST/product/', '') for x in table['dataURL']]
    return table[table['file'].str.endswith('_i2d.fits')].reset_index(drop=True)


def set_files(table, miri_target, nircam_target):
    '''One file per filter for a set, reddest first

    The MIRI image comes from the MIRI target name and the NIRCam images from the NIRCam
    one, they differ. When a filter was observed more than once the latest release is used.
    '''
    instrument = table['instrument_name'].str.split('/').str[0]
    rows = table[((table['target_name'] == miri_target) & (instrument == 'MIRI')) |
                 ((table['target_name'] == nircam_target) & (instrument == 'NIRCAM'))]
    files = []
    for filt, same in rows.groupby('filters'):
        files.append(same.sort_values('t_obs_release').iloc[-1]['file'])
    files = np.array(files)
    return files[np.argsort(-filt_num(files))]


def read_layer(file):
    '''Stream an i2d file from MAST, fill its holes, return image, wcs and headers'''
    with fits.open(MAST_URL + file, use_fsspec=True) as hdul:
        info = hdul[0].header
        header = hdul[1].header
        img = hdul[1].data
    return hole_func_fill(img), WCS(header), header, info


def red_cutout(red, red_wcs, ref_img, ref_wcs):
    '''The part of the reference (bluest) image that holds the red footprint'''
    good = np.isfinite(red) & (red != 0)
    # the MIRI imager i2d also holds the coronagraph strips, keep the main field only
    parts, n_parts = label(good)
    if n_parts > 1:
        good = parts == (np.argmax(np.bincount(parts.ravel())[1:]) + 1)
    rows = np.where(good.any(axis=1))[0]
    cols = np.where(good.any(axis=0))[0]
    if not len(rows) or not len(cols):
        raise Exception('the red image is empty')
    xx, yy = np.meshgrid(cols[[0, -1]], rows[[0, -1]])
    x, y = ref_wcs.world_to_pixel(red_wcs.pixel_to_world(xx.ravel(), yy.ravel()))
    # the footprints are rotated relative to each other, so this is a bounding box
    position = ((x.min() + x.max()) / 2, (y.min() + y.max()) / 2)
    size = (int(y.max() - y.min()), int(x.max() - x.min()))
    return Cutout2D(ref_img, position, size, wcs=ref_wcs, mode='trim')


def color_set(files):
    '''RGB image of the files, reddest first, on the grid of the bluest one'''
    red, red_wcs, _, info = read_layer(files[0])
    blue, blue_wcs, _, _ = read_layer(files[-1])
    cut = red_cutout(red, red_wcs, blue, blue_wcs)
    if min(cut.data.shape) < 10:
        raise Exception(f'no overlap, cutout shape {cut.data.shape}')
    if len(files) == 2:
        # green will be the average of the two, as in astro_jwst_news_color.py
        irgb = [0, len(files) - 1]
    else:
        filt = filt_num(files)
        igreen = np.argmin(np.abs(filt - (filt[0] + filt[-1]) / 2))
        irgb = [0, igreen, len(files) - 1]
    layers = np.zeros((cut.data.shape[0], cut.data.shape[1], 3))
    for jj, ii in enumerate(irgb):
        if ii == 0:
            img, img_wcs = red, red_wcs
        elif ii == len(files) - 1:
            img, img_wcs = blue, blue_wcs
        else:
            img, img_wcs, _, _ = read_layer(files[ii])
        img, _ = reproject_interp((img, img_wcs), output_projection=cut.wcs,
                                  shape_out=cut.data.shape)
        layers[:, :, jj] = np.nan_to_num(img)
    if len(files) == 2:
        layers[:, :, 2] = layers[:, :, 1]
        layers[:, :, 1] = (layers[:, :, 0] + layers[:, :, 2]) / 2
    if layers[:, :, 0].size > MAX_PIX:
        scale = (MAX_PIX / layers[:, :, 0].size) ** 0.5
        layers = transform.resize(layers, [int(layers.shape[0] * scale),
                                           int(layers.shape[1] * scale)])
    for jj in range(3):
        layers[:, :, jj] = level_adjust(layers[:, :, jj], factor=FACTOR)
    return grey_zeros(np.nan_to_num(layers)), info


def save_set_image(row, table=None, out_dir=OUT_DIR, over=False):
    '''Color image of one set, a row of docs/gc_miri_nircam.csv. Returns the jpg path'''
    if table is None:
        table = query_sets()
    name = f"{row['miri_target']}_{row['nircam_target']}_{str(row['set_release'])[:10]}.jpg"
    jpg = f'{out_dir}/{name}'
    if os.path.isfile(jpg) and not over:
        print(f'{name} already exists')
        return jpg
    files = set_files(table, row['miri_target'], row['nircam_target'])
    if len(files) < 2:
        raise Exception(f'only {len(files)} files for the set')
    print(f"{row['miri_target']} + {row['nircam_target']}: "
          f"{', '.join(np.array(filt_num(files)).astype(int).astype(str))}")
    layers, info = color_set(files)
    os.makedirs(out_dir, exist_ok=True)
    plt.imsave(jpg, layers, origin='lower', pil_kwargs={'quality': 95})
    print(f'saved {jpg} {layers.shape[1]}x{layers.shape[0]}, '
          f"PI: {info['PI_NAME']}, program {info['PROGRAM']}")
    return jpg


if __name__ == '__main__':
    args = sys.argv[1:]
    over = 'over' in [a.lower() for a in args]
    limit = 0
    target_arg = None
    for a in args:
        if a.isdigit():
            limit = int(a)
        elif a.upper().startswith('GC_'):
            target_arg = a.upper()
    sets = pd.read_csv(SETS_CSV)
    sets = sets[sets['miri_obs_date'].notna() & sets['nircam_obs_date'].notna()]
    if target_arg:
        sets = sets[sets['miri_target'] == target_arg]
        if not len(sets):
            raise Exception(f'{target_arg} is not a complete set in {SETS_CSV}')
    print(f'{len(sets)} complete sets in {SETS_CSV}')
    table = query_sets()
    done = 0
    for _, set_row in sets.iterrows():
        jpg = f"{OUT_DIR}/{set_row['miri_target']}_{set_row['nircam_target']}_" \
              f"{str(set_row['set_release'])[:10]}.jpg"
        if os.path.isfile(jpg) and not over:
            print(f'{jpg.split("/")[-1]} already exists')
            continue
        try:
            save_set_image(set_row, table=table, over=over)
        except Exception as e:
            print(f"failed color image for {set_row['miri_target']}: {e}")
            continue
        done += 1
        if limit and done >= limit:
            print(f'stopping after {limit} images')
            break
    print(f'{done} new color images in {OUT_DIR}')
