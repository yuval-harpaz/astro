import os
from astro_utils import *
import astropy
from easygui import *
# from astropy.time import Time
# from astro_list_ngc import choose_fits, make_thumb, ngc_html_thumb
# from glob import glob
# from astro_ngc_align import add_crval_to_logs




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


 
# dialog
text = "Selected one target_name"
title = "astro"
tgt = choicebox(text, title, sorted(new_targets))
rows = np.where(science['target_name'] == tgt)[0]
files = science['dataURL'].values[rows]
files = [f.split('/')[-1] for f in files]
if os.path.isdir(drive):
    destination_folder = drive+'data/' + tgt
else:
    destination_folder = 'data/' + tgt
download_fits_files(files, destination_folder=destination_folder, wget=False)




##




