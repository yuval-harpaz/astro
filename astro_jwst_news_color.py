'''
The code creates a web page to view the latest JWST images
See here: https://bsky.app/profile/astrobotjwst.bsky.social
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
from astro_ngc_preview_2025 import post_image
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
deband = False
bsky = True
if len(chosen_targets) == 0:
    print('no new targets for color processing')
else:
    for target in chosen_targets:
        # target = new_targets[np.where(t_obs_release == sec_latest)[0][0]]
        row1 = np.where(science['target_name'].values == target)[0][0]
        mast_url = 'https://mast.stsci.edu/portal/Download/file/JWST/product/'
        files = science['dataURL'][science['target_name'] == target].values
        files = [x.replace('mast:JWST/product/', '') for x in files]
        files = np.array([x for x in files if 'niriss' not in x])
        max_t_release = max(science['t_obs_release'][science['target_name'] == target].values)

        crval = []
        for file in files:
            try:
                with fits.open(mast_url+file, use_fsspec=True) as hdul:
                    header = hdul[1].header
                    crval.append([header['CRVAL1'], header['CRVAL2']])
            except Exception as e:
                print(f"Error processing {file}: {e}")
                crval.append([np.nan, np.nan])
        label = cluster_coordinates(crval)
        groups = []
        for g in range(max(label)+1):
            groups.append(files[label == g])
        # groups, _ = overlap(files)
        groups = [g for g in groups if len(g) > 2]
        for group_files in groups:
            filt = filt_num(group_files)
            order = np.argsort(-filt)
            group_files = np.array(group_files)[order]
            filt = filt[order]
            # group_files = files[group]
            # download red first
            igreen = np.argmin(np.abs(filt - (filt[0] + filt[-1])/2))
            irgb = [0, igreen, len(group_files)-1]
            if group_files[irgb[2]] in df['blue'].values and group_files[irgb[0]] in df['red'].values:
                print(f"{target} file already used as blue:  {group_files[irgb[2]]} (also red was used)")  # sometimes already fails to detect extra MIRI with no overlap
            else:
                goon = False
                try:
                    for jj, ii in enumerate(irgb):
                        fn = group_files[ii]
                        # download_fits_files([fn], 'data/tmp')
                        if ii == 0:
                            # hdu0 = fits.open('data/tmp/' + fn)
                            with fits.open(mast_url+fn, use_fsspec=True) as hdul:
                                hdr0 = hdul[1].header
                                img = hdul[1].data
                                info = hdul[0].header
                            # hdu0 = fits.open(mast_url+fn, )
                            # img = hdu0[1].data
                            img = hole_func_fill(img)
                            if deband:
                                img = deband_layer(img, func=np.percentile)
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
                            if deband:
                                hdu.data = deband_layer(hdu.data, func=np.percentile)
                            img, _ = reproject_interp(hdu, hdr0)
                            img = transform.resize(img, [layers.shape[0], layers.shape[1]])
                            # hdu.close()
                        # os.remove('data/tmp/' + fn)
                        img = level_adjust(img, factor=2)
                        layers[:, :, jj] = img
                    goon = True
                except:
                    print(f'failed download or process color images for {target}')
                new_row = [target, max_t_release, group_files[irgb[0]], group_files[irgb[1]], group_files[irgb[2]], 'failed']
                if goon:
                    try:
                        layers[np.isnan(layers)] = 0
                        plt.imsave('data/tmp/tmprs.jpg', layers, origin='lower', pil_kwargs={'quality':95})
                        # new_targets = ', '.join(np.unique(new['target_name']))
                        filt_str = ', '.join(filt[irgb].astype(int).astype(str))
                        toot = f"\U0001F916 image processing for #JWST \U0001F52D data ({target}). RGB Filters: {filt_str}"
                        toot = toot + f"\nPI: {info['PI_NAME']}, program {info['PROGRAM']}. CRVAL: {np.round(hdr0['CRVAL1'], 6)}, {np.round(hdr0['CRVAL2'], 6)}"
                        toot = toot+'\nCredits: NASA, ESA, CSA, STScI.'
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
                        print(f'failed preparing toot for {target}')
                        goon = False
                if goon:
                    # success = ''
                    post = post_image(txt, 'data/tmp/tmprs.jpg', mastodon=True, bluesky=bsky)
                    if 'masto' in post.keys():
                        url = post['masto']['url']+';'
                    else:
                        url = ''
                    if 'bsky' in post.keys():
                        url = url + 'https://bsky.app/profile/astrobotjwst.bsky.social/post/'+post['bsky']['uri'].split('/')[-1] + ';'
                    url = url[:-1]
                    # if bsky:
                    #     try:
                    #         with open('data/tmp/tmprs.jpg', 'rb') as f:
                    #             img_data = f.read()
                    #         alt = "Automatic color preview of JWST data"
                    #         post = blient.send_image(text=boot, image=img_data, image_alt=alt)
                    #         success += 'bsky;'
                    #         print('boot image')
                    #     except:
                    #         print(f'failed bluesky color image post for {target}')
                    # try:
                    #     metadata = masto.media_post("data/tmp/tmprs.jpg", "image/jpeg")
                    #     _ = masto.status_post(toot, media_ids=metadata["id"])
                    #     success += 'masto;'
                    #     print('toot image')
                    # except:
                    #     print(f'failed mastodon color image post for {target}')
                    if url:
                        # success = success[:-1]
                        new_row[-1] = url
                df.loc[len(df)] = new_row
                df.to_csv('docs/bot_color_posts.csv', index=False)
        print('done auto color processing for '+target)
    # imgrs = resize_with_padding(img)
    # plt.imsave('tmprs.jpg', imgrs, cmap='gray')
    # size = os.path.getsize('tmp.jpg')
    # mb2 = 2*10**6  # mastodon allows 2MB
## create a list of download links
