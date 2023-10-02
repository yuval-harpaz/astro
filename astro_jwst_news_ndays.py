'''
The code reates a web page to view the latest JWST images
https://yuval-harpaz.github.io/astro/news_by_date.html
@Author: Yuval Harpaz
'''
# requires pandas as well, no need to import
import astropy
# import matplotlib.pyplot as plt
from astroquery.mast import Observations
import numpy as np
from mastodon_bot import connect_bot
# from skimage.transform import resize
import os
from astro_list_ngc import social, credits
from astro_utils import download_obs, auto_plot
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
calib = False
# for calib in [False, True]:
# if calib:
# suf = '_calib'
# tit = 'calibration'
# tbl = table[calibration]
# other = '. see also <a href="https://yuval-harpaz.github.io/astro/news_by_date.html" target="_blank">science images</a>'
# download = ', <a href="https://yuval-harpaz.github.io/astro/downloads_by_date_calib.html" target="_blank">downloads</a>'
# else:
suf = ''
tit = 'science'
tbl = table[~calibration]
# other = '. see also <a href="https://yuval-harpaz.github.io/astro/news_by_date_calib.html" target="_blank">calibration images</a>'
download = ', <a href="https://yuval-harpaz.github.io/astro/downloads_by_date.html" target="_blank">downloads</a>'
html_name = 'docs/news_by_date' + suf + '.html'
with open(html_name, 'r') as fid:
    last = fid.read()
first_image = last.index('<img')
last_date = last[last.index('newest:')+8:last.index('\n')-1]
if len(tbl) > 0:
    new_date = astropy.time.Time(tbl['t_obs_release'].iloc[0], format='mjd').utc.iso
    page = '<!DOCTYPE html><!newest: '+new_date+'>\n<html>\n<head>\n' \
           '  <link rel="stylesheet" href="styles.css">' \
           '  <title>JWST latest release</title></title><link rel="icon" type="image/x-icon" href="camelfav.ico" />\n  <style>\n   img {\n      max-width: 19vw; /* Limit image width to P% of viewport width */\n      height: auto; /* Maintain aspect ratio */\n    }\n  </style>\n</head>\n<body>'
    page = page + '<h1>JWST ' + tit + ' images by release date (' + str(n) + ' days)</h1>'
    # add = '<a href="https://github.com/yuval-harpaz/astro/blob/main/ngc.csv" target="_blank"> table</a>\n    '
    add = download # + other
    page = page + social(add=add)
    #<h2>by <a href="https://twitter.com/yuvharpaz" target="_blank">@yuvharpaz</a>, <a href="https://github.com/yuval-harpaz/astro/blob/main/astro_jwst_news_ndays.py" target="_blank"> code</a>' + download + other + '. Follow the <a href="https://botsin.space/@astrobot_jwst" target="_blank">bot</a>.<br>'
    date_prev = ''  # add date every time there is a new dataset
    # print('making html')
    for iimg in range(len(tbl)):  # min([len(tbl), n])):
        time = astropy.time.Time(tbl['t_obs_release'].iloc[iimg], format='mjd').utc.iso
        date = time[:16]
        if date != date_prev:
            page = page + '\n<br>' +date + '<br>\n'
        jpg = tbl['jpegURL'].iloc[iimg]
        if type(jpg) == str:
            jpg = jpg.replace('mast:JWST/product/', '')
        else:
            print(f'strange jpg path {jpg}')
            jpg = ''
        desc = 'title: ' + tbl['obs_title'].iloc[iimg] + '\n' + \
               'target: ' + tbl['target_name'].iloc[iimg] + '\n' + \
               'proposal: ' + str(tbl['proposal_id'].iloc[iimg]) + '\n' + \
               jpg + '\n' + date
        date_prev = date
        page = page + '\n<img src="https://mast.stsci.edu/portal/Download/file/JWST/product/' + jpg + f'" title="{desc}">'
    page = page + cred + '\n</body>\n</html>\n'
    with open(html_name, "w") as text_file:
        text_file.write(page)
    first_image = page.index('https://mast')
    # new_date = page[first_image - 16:first_image - 6]
    if new_date[:16] > last_date[:16] and not calib:
        for tblrow in range(len(tbl)):
            nd = astropy.time.Time(tbl['t_obs_release'].iloc[tblrow], format='mjd').utc.iso[:16]
            if nd == last_date[:16]:
                print(f'update {tblrow} rows')
                break
        # last = np.where(tbl['target_name'] == df_prev['target_name'][0])[0][-1]
        tgt_list = list(np.unique(tbl.iloc[:tblrow]['target_name']))
        tgts = ','
        tgts = tgts.join(tgt_list)
        print(f'debug info ndays:\n'
              f'targets: {tgts}\n'
              f'table row: {tblrow}\n'
              f'new date: {new_date[:16]}, prev date: {last_date[:16]}')
        toot = f'New {tit} images ({tgts}), take a look at https://yuval-harpaz.github.io/astro/{html_name[5:]}'
        # toot = 'new ' + tit + ' images at https://yuval-harpaz.github.io/astro/' + html_name[5:]
        tbl_new = tbl.iloc[:tblrow, :]
        tbl_new.to_csv('tmp_new.csv', index=False)
        masto, loc = connect_bot()
        failed = False
        for tgt in tgt_list:
            try:
                #TODO: if one file, adjust levels and toot
                download_obs(tbl_new[tbl_new['target_name'] == tgt])
                auto_plot(tgt, exp='*.fits', png='tmp.jpg', pkl=False, resize=True, method='rrgggbb',
                          plot=False, fill=False, deband=False, adj_args={'factor': 2}, blc=False)
                metadata = masto.media_post("tmp.jpg", "image/jpeg")
                masto.status_post('New public JWST data: '+tgt, media_ids=metadata["id"])
                print('toot image')
                if loc == 'github':
                    os.system('rm -rf data/'+tgt)
            except:
                print('failed '+tgt)
                failed = True
        if failed:
            masto.status_post(toot)
            print('toot')
            # a = os.system('wget -O tmp.jpg '+page[first_image:page.index('.jpg')+4] + '>/dev/null 2>&1')
            # if a == 0:
            #     img = plt.imread('tmp.jpg')
            #     size = os.path.getsize('tmp.jpg')
            #     mb2 = 2*10**6  # mastodon allows 2MB
            #     if size >= mb2:
            #         ratio = mb2/size
            #         img = resize(img, (int(ratio**0.5*img.shape[0]), int(ratio**0.5*img.shape[1])))
            #         plt.imsave('tmp.jpg', img, cmap='gray')
            #     try:
            #         metadata = masto.media_post("tmp.jpg", "image/jpeg")
            #         masto.status_post(toot, media_ids=metadata["id"])
            #         print('toot image')
            #     except:
            #         try:
            #             metadata = masto.media_post("tmp.jpg", "image/jpeg")
            #             ratio = 1500/np.max(img.shape)
            #             img = resize(img, (int(ratio * img.shape[0]), int(ratio * img.shape[1])))
            #             plt.imsave('tmp.jpg', img, cmap='gray')
            #             metadata = masto.media_post("tmp.jpg", "image/jpeg")
            #             masto.status_post(toot, media_ids=metadata["id"])
            #             print('toot image')
            #         except:
            #             masto.status_post(toot)
            #             print('toot')

print('wrote image preview')
## create a list of download links
# for calib in [False, True]:
#     if calib:
# suf = '_calib'
# tit = 'calibration'
# tbl = table[calibration]
# other = '. see also <a href="https://yuval-harpaz.github.io/astro/downloads_by_date.html" target="_blank">science images</a>'
# else:
suf = ''
tit = 'science'
tbl = table[~calibration]
# other = '. see also <a href="https://yuval-harpaz.github.io/astro/downloads_by_date_calib.html" target="_blank">calibration images</a>'
if len(tbl) > 0:
    page = '<!DOCTYPE html>\n<html>\n<head>\n  <title>JWST latest release</title></title><link rel="icon" type="image/x-icon" href="camelfav.ico" />\n  <style>\n   img {\n      max-width: 19vw; /* Limit image width to P% of viewport width */\n      height: auto; /* Maintain aspect ratio */\n    }\n  </style>\n</head>\n<body>'
    page = page + '<h1>JWST ' + tit + ' images by release date (' + str(n) + \
            ' days)</h1><h2>by <a href="https://twitter.com/yuvharpaz" target="_blank">@yuvharpaz</a>, <a href="https://github.com/yuval-harpaz/astro/blob/main/astro_jwst_news_ndays.py" target="_blank"> code</a><br>'
    date_prev = ''
    for iimg in range(len(tbl)):  # min([len(tbl), n])):
        time = astropy.time.Time(tbl['t_obs_release'].iloc[iimg], format='mjd').utc.iso
        date = time[:16]
        if date != date_prev:
            page = page + '\n<br>' +date + '<br>\n'
        target = tbl['dataURL'].iloc[iimg].replace('mast:JWST/product/', '')
        tname = tbl['target_name'].iloc[iimg]
        date_prev = date
        page = page + f'\n{tname} <a href="https://mast.stsci.edu/portal/Download/file/JWST/product/{target}"' \
                      f' target="_blank">{target}</a><br>'
    page = page + '\n</body>\n</html>\n'
    with open('docs/downloads_by_date'+suf+'.html', "w") as text_file:
        text_file.write(page)
print('done')
