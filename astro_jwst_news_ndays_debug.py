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
from astro_utils import resize_with_padding
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
tbl = table[~calibration]
science = tbl.copy()

keep = ['obsid', 'proposal_id', 'proposal_pi', 'target_name', 'instrument_name', 'obs_title', 'jpegURL', 't_max', 't_obs_release']
latest = pd.DataFrame(columns=keep)

for col in keep:
    latest[col] = science[col]
latest = latest.reset_index(drop=True)
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
            new.at[row, col] = astropy.time.Time(float(new[col][row]), format='mjd').utc.iso
    latest_new = pd.concat([prev, new])
    latest_new = latest_new.sort_values('t_obs_release', ignore_index=True, ascending=False)
    latest.to_csv('docs/latest.csv', index=False)
    print('saved new latest')
## create a list of download links
