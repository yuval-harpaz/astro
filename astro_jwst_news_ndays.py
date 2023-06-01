'''
The code reates a web page to view the latest JWST images
https://yuval-harpaz.github.io/astro/news_by_date.html
@Author: Yuval Harpaz
'''
# requires pandas as well, no need to import
import astropy
from astroquery.mast import Observations


# n days to look back for new releases
n = 14
print(f'reading {n} days')
end_time = astropy.time.Time.now().mjd
start_time = end_time - n
## Query MAST database for level 3 JWST images
table = Observations.query_criteria(obs_collection="JWST",
                                    t_obs_release=[start_time, end_time],
                                    calib_level=3,
                                    dataproduct_type="image")
## make sure images are public, science, no Calibration in title
if len(table) > 0:
    table = table[table['intentType'] == 'science']
    if len(table) > 0:
        table = table[table['dataRights'] == 'PUBLIC']
        if len(table) > 0:
            tdf = table.to_pandas()
            table = table[list(~tdf['obs_title'].str.contains("alibration"))]
# Rewrite the web page if there's any data from the last 14 days. Set titles to show description when
# hover over images
if len(table) == 0:
    raise Exception('no new images')
table = table.to_pandas()
table = table.sort_values('t_obs_release', ascending=False, ignore_index=True)
page = '<!DOCTYPE html>\n<html>\n<head>\n  <title>JWST latest release</title>\n  <style>\n   img {\n      max-width: 19vw; /* Limit image width to P% of viewport width */\n      height: auto; /* Maintain aspect ratio */\n    }\n  </style>\n</head>\n<body>'
page += '<h1>JWST science images by release date (' + str(n) + ' days)</h1><h2>by <a href="https://twitter.com/yuvharpaz" target="_blank">@yuvharpaz</a>, <a href="https://github.com/yuval-harpaz/astro/blob/main/astro_jwst_news_ndays.py" target="_blank"> code</a><br>'
date_prev = ''
print('making html')
for iimg in range(len(table)):  # min([len(table), n])):
    time = astropy.time.Time(table['t_obs_release'][iimg], format='mjd').utc.iso
    date = time[:10]
    if date != date_prev:
        page = page + '\n<br>' +date + '<br>\n'
    jpg = table['jpegURL'][iimg].replace('mast:JWST/product/', '')
    desc = 'title: ' + table['obs_title'][iimg] + '\n' + \
           'target: ' + table['target_name'][iimg] + '\n' + \
           'proposal: ' + str(table['proposal_id'][iimg]) + '\n' + \
           jpg + '\n' + time[:16]
    date_prev = date
    page = page + '\n<img src="https://mast.stsci.edu/portal/Download/file/JWST/product/' + jpg + f'" title="{desc}">'
page = page + '\n</body>\n</html>\n'
with open('docs/news_by_date.html', "w") as text_file:
    text_file.write(page)

