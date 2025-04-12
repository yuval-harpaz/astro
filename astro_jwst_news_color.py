'''
The code reates a web page to view the latest JWST images
https://yuval-harpaz.github.io/astro/news_by_date.html
@Author: Yuval Harpaz
'''
# requires pandas as well, no need to import
import astropy
import matplotlib.pyplot as plt
from astroquery.mast import Observations
import numpy as np
from mastodon_bot import connect_bot
from skimage.transform import resize
import os
from astro_list_ngc import social, credits
from atproto import Client as Blient
from atproto import client_utils
# from astro_utils import resize_to_under_2mb, download_fits_files, filt_num
from astro_utils import *
import pandas as pd
import requests
# n days to look back for new releases
n = 7
print(f'reading {n} days')
end_time = astropy.time.Time.now().mjd
start_time = end_time - n
## Query MAST database for level 3 JWST images
args = {'obs_collection': "JWST",
        'calib_level': 3,
        'dataRights': 'public',
        # 'intentType': 'science',
        'dataproduct_type': "image"}

# search images by observation date and by release date
table_release = Observations.query_criteria(t_obs_release=[start_time, end_time], **args)
table_min = Observations.query_criteria(t_min=[start_time, end_time], **args)
table = table_min.to_pandas().merge(table_release.to_pandas(), how='outer')
# Rewrite the web page if there's any data from the last 14 days. Set titles to show description when
# hover over images
if len(table) == 0:
    raise Exception('no new images')
# table = table.to_pandas()
table = table.sort_values('t_obs_release', ascending=False, ignore_index=True)
calibration = np.asarray((table['obs_title'].str.contains("alibration")) | (table['intentType'] != 'science'))
cred = credits()
science = table[~calibration]
science = science.reset_index(drop=True)
for col in ['t_obs_release', 't_max']:
    for row in range(len(science)):
        science.at[row, col] = astropy.time.Time(science[col][row], format='mjd').utc.iso

new_targets, n_targets = np.unique(science['target_name'], return_counts=True)
t_obs_release = []
for itarget in range(len(n_targets)):
    rows = np.where(science['target_name'] == new_targets[itarget])[0]
    t_obs_release.append(np.max(science['t_obs_release'].values[rows]))
t_obs_release = np.array(t_obs_release)

if not os.path.isfile('docs/latest.csv'):
    os.chdir(os.environ['HOME']+'/astro')
    # raise Exception('am I in astro?')
if not os.path.exists('data/tmp'):
    os.makedirs('data/tmp')

df = pd.read_csv('docs/bot_color_posts.csv')
min_filt = np.zeros(len(new_targets))
max_filt = np.zeros(len(new_targets))
already = np.zeros(len(new_targets), bool)
for itarget in range(len(new_targets)):
    cand = new_targets[itarget]
    icand = np.where(df['target_name'].values == cand)[0]
    if len(icand) > 0:  #compare filters of the current target to last occurance of same target
        icand = icand[-1]
        target_rows = np.where(science['target_name'].values == cand)[0]
        obs_filt = filt_num(science['dataURL'].values[target_rows])
        if min(obs_filt) >= filt_num([df['blue'].values[icand]])[0] and \
           max(obs_filt) <= filt_num([df['red'].values[icand]])[0]:
            already[itarget] = True

# prev_target = df['target_name'].values[-1]


# Exclude latest, Data may still be coming
# not_prev = new_targets != prev_target
# not_latest = np.array(t_obs_release < max(t_obs_release))
# include = not_latest & (n_targets > 2) & (n_targets < 15) & not_prev
include = ~already & (n_targets > 2) & (n_targets < 15)
chosen_targets = new_targets[include]
# sec_latest = max(t_obs_release[include])
for target in chosen_targets:
    # target = new_targets[np.where(t_obs_release == sec_latest)[0][0]]
    row1 = np.where(science['target_name'].values == target)[0][0]
    obsid = int(science['obsid'][row1])
    mast_url = 'https://mast.stsci.edu/portal/Download/file/JWST/product/'
    files = science['dataURL'][science['target_name'] == target].values
    files = [x.replace('mast:JWST/product/', '') for x in files]
    filt = filt_num(files)
    order = np.argsort(filt)[::-1]
    files = np.array(files)[order]
    filt = filt[order]
    # download red first
    igreen = np.argmin(np.abs(filt - (filt[0] + filt[-1])/2))
    irgb = [0, igreen, len(files)-1]
    if files[irgb[2]] in df['blue'].values:
        raise Exception(f"{target} file laready used as blue:  {files[irgb[2]]}")
    goon = False
    try:
        for jj, ii in enumerate(irgb):
            fn = files[ii]
            # download_fits_files([fn], 'data/tmp')
            if ii == 0:
                # hdu0 = fits.open('data/tmp/' + fn)
                with fits.open(mast_url+fn, use_fsspec=True) as hdul:
                    hdr0 = hdul[1].header
                    img = hdul[1].data
                # hdu0 = fits.open(mast_url+fn, )
                # img = hdu0[1].data
                img = hole_func_fill(img)
                img = resize_to_under_1mp(img)
                layers = np.zeros((img.shape[0], img.shape[1], 3))
                # hdr0 = hdu0[1].header
                # hdu0.close()
            else:
                with fits.open(mast_url+fn, use_fsspec=True) as hdul:
                    hdr = hdul[1].header
                    img = hdul[1].data
                hdu = fits.ImageHDU()
                hdu.data = img
                hdu.header = hdr
                # hdu = fits.open('data/tmp/' + fn)
                hdu.data = hole_func_fill(hdu.data)
                img, _ = reproject_interp(hdu, hdr0)
                img = transform.resize(img, [layers.shape[0], layers.shape[1]])
                # hdu.close()
            # os.remove('data/tmp/' + fn)
            img = level_adjust(img, factor=2)
            layers[:, :, jj] = img
        goon = True
    except:
        print('failed download or process color images')
    
    new_row = [target,sec_latest, files[irgb[0]], files[irgb[1]], files[irgb[2]], 'failed', obsid]
    if goon:
        try:
            layers[np.isnan(layers)] = 0
            plt.imsave('data/tmp/tmprs.jpg', layers, origin='lower', pil_kwargs={'quality':95})
            # new_targets = ', '.join(np.unique(new['target_name']))
            filt_str = ', '.join(filt[irgb].astype(int).astype(str))
            toot = f"Bot image processing for NASA / STScI #JWST \U0001F52D data ({target}). RGB Filters: {filt_str}"
            blient = Blient()
            blient.login(os.environ['Bluehandle'], os.environ['Blueword'])
            boot = client_utils.TextBuilder()
            blue = True
            blim = 250  # should be 300 limit but failed once
            txt = toot
            if len(txt) > blim:
                txt = txt[:blim]
            boot.text(txt)
            masto, loc = connect_bot()
        except:
            print('failed preparing toot')
            goon = False
    if goon:
        success = ''
        try:
            with open('data/tmp/tmprs.jpg', 'rb') as f:
                img_data = f.read()
            alt = "Automatic color preview of JWST data"
            post = blient.send_image(text=boot, image=img_data, image_alt=alt)
            success += 'bsky;'
            print('boot image')
        except:
            print('failed bluesky color image post')
        # try:
        #     metadata = masto.media_post("data/tmp/tmprs.jpg", "image/jpeg")
        #     _ = masto.status_post(toot, media_ids=metadata["id"])
        #     success += 'masto;'
        #     print('toot image')
        # except:
        #     print('failed mastodon color image post')
        if success:
            success = success[:-1]
            new_row[-1] = success
    df.loc[len(df)] = new_row
    df.to_csv('docs/bot_color_posts.csv', index=False)
print('done auto color processing')
    # imgrs = resize_with_padding(img)
    # plt.imsave('tmprs.jpg', imgrs, cmap='gray')
    # size = os.path.getsize('tmp.jpg')
    # mb2 = 2*10**6  # mastodon allows 2MB
## create a list of download links
