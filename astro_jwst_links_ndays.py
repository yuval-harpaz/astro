'''
The code reates a web page to view the latest JWST images
https://yuval-harpaz.github.io/astro/news_by_date.html
@Author: Yuval Harpaz
'''
# requires pandas as well, no need to import
import astropy
from astroquery.mast import Observations
import numpy as np

# n days to look back for new releases
n = 14
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

for calib in [False, True]:
    if calib:
        suf = '_calib'
        tit = 'calibration'
        tbl = table[calibration]
        other = '. see also <a href="https://yuval-harpaz.github.io/astro/downloads_by_date.html" target="_blank">science images</a>'
    else:
        suf = ''
        tit = 'science'
        tbl = table[~calibration]
        other = '. see also <a href="https://yuval-harpaz.github.io/astro/downloads_by_date_calib.html" target="_blank">calibration images</a>'
    if len(tbl) > 0:
        page = '<!DOCTYPE html>\n<html>\n<head>\n  <title>JWST latest release</title>\n  <style>\n   img {\n      max-width: 19vw; /* Limit image width to P% of viewport width */\n      height: auto; /* Maintain aspect ratio */\n    }\n  </style>\n</head>\n<body>'
        page = page + '<h1>JWST ' + tit + ' images by release date (' + str(n) + \
                ' days)</h1><h2>by <a href="https://twitter.com/yuvharpaz" target="_blank">@yuvharpaz</a>, <a href="https://github.com/yuval-harpaz/astro/blob/main/astro_jwst_news_ndays.py" target="_blank"> code</a>' + other + '<br>'
        date_prev = ''
        # print('making html')
        for iimg in range(len(tbl)):  # min([len(tbl), n])):
            time = astropy.time.Time(tbl['t_obs_release'].iloc[iimg], format='mjd').utc.iso
            date = time[:10]
            if date != date_prev:
                page = page + '\n<br>' +date + '<br>\n'
            target = tbl['dataURL'].iloc[iimg].replace('mast:JWST/product/', '')
            # desc = 'title: ' + tbl['obs_title'].iloc[iimg] + '\n' + \
            #        'target: ' + tbl['target_name'].iloc[iimg] + '\n' + \
            #        'proposal: ' + str(tbl['proposal_id'].iloc[iimg]) + '\n' + \
            #        jpg + '\n' + time[:16]
            date_prev = date
            page = page + f'\n<a href="https://mast.stsci.edu/portal/Download/file/JWST/product/{target}"' \
                          f' target="_blank">{target}</a><br>'
        page = page + '\n</body>\n</html>\n'
        with open('docs/downloads_by_date'+suf+'.html', "w") as text_file:
            text_file.write(page)


