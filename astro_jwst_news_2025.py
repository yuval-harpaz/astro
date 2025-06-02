'''
The code creates a web page to view the latest JWST images
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
from astro_utils import resize_to_under_2mb
import pandas as pd
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
# science = tbl.copy()
##
url = 'https://yuval-harpaz.github.io/astro/jwst_latest_release.html'
mast_url = 'https://mast.stsci.edu/portal/Download/file/'
keep = ['obsid', 'proposal_id', 'proposal_pi', 'target_name', 'instrument_name', 'obs_title', 'jpegURL', 'dataURL', 't_max', 't_obs_release']
latest = pd.DataFrame(columns=keep)

for col in keep:
    latest[col] = science[col]
latest = latest.reset_index(drop=True)
nanjpeg = np.where(latest['jpegURL'].isnull())[0]
if len(nanjpeg) > 0:
    for ii in nanjpeg:
        if '.fits' in str(latest['dataURL'][ii]):
            repurl = latest['dataURL'][ii].replace('.fits','.jpg')
        else:
            repurl = 'docs/broken.jpg'
        latest.at[ii, 'jpegURL'] = repurl

prev = pd.read_csv('docs/latest.csv')
previd = prev['obsid'].values
inew = []
for x in range(len(latest)):
    if int(latest['obsid'][x]) not in previd:
        inew.append(x)
# inew = [x for x in  if int(latest['obsid'][x]) not in previd]
if len(inew) > 0:
    new = latest.iloc[inew]
    new = new.reset_index(drop=True)
    for col in ['t_obs_release', 't_max']:
        for row in range(len(new)):
            new.at[row, col] = astropy.time.Time(new[col][row], format='mjd').utc.iso
    prev_new = pd.concat([prev, new])
    prev_new = prev_new.sort_values('t_obs_release', ignore_index=True, ascending=False)
    prev_new.to_csv('docs/latest.csv', index=False)
    print('saved new latest')
    try:
        new_targets = ', '.join(np.unique(new['target_name']))
        toot = f'New #JWST \U0001F52D data release for target names: {new_targets}.\nCredit: NASA, ESA, CSA, STScI.\nTake a look at {url}'
        blient = Blient()
        blient.login(os.environ['Bluehandle'], os.environ['Blueword'])
        boot = client_utils.TextBuilder()
        blue = True
        blim = 250  # should be 300 limit but failed once
        if 'https' in toot:
            txt = toot[:toot.index('https')]
            if len(txt) > blim:
                txt = txt[:blim]
            boot.text(txt)
            boot.link('news_by_date.html', toot[toot.index('https'):])
        else:
            txt = toot
            if len(txt) > blim:
                txt = txt[:blim]
            boot.text(txt)
        first_image_url = new['jpegURL'][0].replace('mast:', mast_url)
        err = os.system(f"wget -O tmp.jpg {first_image_url} >/dev/null 2>&1")
        if err:
            got_image = False
            toot_text_only_b = True
            toot_text_only_m = True
        else:
            got_image = True
            toot_text_only_b = False
            toot_text_only_m = False
        masto, loc = connect_bot()
        if got_image:
            img = plt.imread('tmp.jpg')
            print('resizing')
            _ = resize_to_under_2mb(img, max_size_mb=1)
            try:
                jpg_fn = first_image_url.split('/')[-1]
                with open('tmprs.jpg', 'rb') as f:
                    img_data = f.read()
                assert(jpg_fn in new['jpegURL'][0])
                tbl_row = np.where(new['jpegURL'].str.contains(jpg_fn))[0]
                if len(tbl_row) == 1:
                    alt = f"{jpg_fn}\n{new.iloc[tbl_row[0]]['target_name']}\n{new.iloc[tbl_row[0]]['obs_title']}"
                post = blient.send_image(text=boot, image=img_data, image_alt=alt)
            except:
                print('failed bluesky image post')
                toot_text_only_b = True

            try:
                metadata = masto.media_post("tmprs.jpg", "image/jpeg")
                _ = masto.status_post(toot, media_ids=metadata["id"])
                print('toot image')
            except:
                print('failed mastodon image post')
                toot_text_only_m = True
        if toot_text_only_b:
            try:
                post = blient.send_post(boot)
                print('boot')
            except:
                print('failed bluesky text post')
        if toot_text_only_m:
            try:
                _ = masto.status_post(toot)
                print('toot')
            except:
                print('failed mastodon text post')
    except:
        print('something or other failed')
print('done news updates')
    # imgrs = resize_with_padding(img)
    # plt.imsave('tmprs.jpg', imgrs, cmap='gray')
    # size = os.path.getsize('tmp.jpg')
    # mb2 = 2*10**6  # mastodon allows 2MB
## create a list of download links
