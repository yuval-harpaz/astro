import os
from astro_utils import *
import astropy
from easygui import *
from astroquery.mast import Observations
import numpy as np
import astropy
import sys
# from astropy.time import Time
# from astro_list_ngc import choose_fits, make_thumb, ngc_html_thumb
# from glob import glob
# from astro_ngc_align import add_crval_to_logs

def download_target_by_name(target):

    args = {'obs_collection': "JWST",
            'calib_level': 3,
            'dataRights': 'public',
            # 'intentType': 'science',
            'target_name': target,
            'dataproduct_type': "image"}
    obs_table = Observations.query_criteria(**args)
    # search images by observation date and by release date
    # table_release = Observations.query_criteria(t_obs_release=[start_time, end_time], **args)
    # table_min = Observations.query_criteria(t_min=[start_time, end_time], **args)
    # table = table_min.to_pandas().merge(table_release.to_pandas(), how='outer')
    # Rewrite the web page if there's any data from the last 14 days. Set titles to show description when
    # hover over images
    # obs_table = Observations.query_criteria(obs_collection='JWST', target_name=target)
    if len(obs_table) == 0:
        raise Exception('no images for '+target)
    table = obs_table.to_pandas()
    table = table.sort_values('t_obs_release', ascending=False, ignore_index=True)
    calibration = np.asarray((table['obs_title'].str.contains("alibration")) | (table['intentType'] != 'science'))
    # cred = credits()
    science = table[~calibration]
    science = science.reset_index(drop=True)
    for col in ['t_obs_release', 't_max']:
        for row in range(len(science)):
            science.at[row, col] = astropy.time.Time(science[col][row], format='mjd').utc.iso
    t_obs_release = np.array([t[:10] for t in science['t_obs_release'].values])
    times, n_times = np.unique(t_obs_release, return_counts=True)
    # dialog
    if len(times) == 1:
        download = buttonbox(
            "Only one t_obs_release, "+times[0]+". Download?",
            title="Astro",
            choices=["y", "n"],          # single button
            default_choice="y"      # (redundant here, but explicit)
        )
        if str(download) == 'y':
            tgt = times[0]
        else:
            print('n chosen, abort')
            return
    else:
        text = "Selected one t_obs_release"
        title = "astro"
        tgt = choicebox(text, title, sorted(times))
    rows = np.where(t_obs_release == tgt)[0]
    files = science['dataURL'].values[rows]
    files = [f.split('/')[-1] for f in files]
    if os.path.isdir(drive):
        destination_folder = drive+'data/' + target
    else:
        destination_folder = 'data/' + target
    download_fits_files(files, destination_folder=destination_folder, wget=False)


if __name__ == '__main__':
    download_target_by_name(sys.argv[1])

##




