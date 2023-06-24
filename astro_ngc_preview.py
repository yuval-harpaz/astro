# import pandas as pd
import os.path

from astro_utils import *
from astropy.time import Time
# from mastodon_bot import connect_bot
from astro_list_ngc import list_ngc, ngc_html, choose_fits
from glob import glob
from skimage.transform import resize
from pyongc import ongc
##
df = pd.read_csv('ngc.csv', sep=',')
# df = list_ngc()
# df.to_csv('ngc.csv', sep=',', index=False)
# ngc_html()
# masto, loc = connect_bot()
##
# loc == 'local'
for row in range(3):
    pkl = True
    tgt = df['target_name'][row]
    drive = '/media/innereye/My Passport/Data/JWST/'
    if os.path.isdir(drive):
        os.chdir(drive)
    if not os.path.isdir('data'):
        raise Exception('where are we?')
        # os.system('mkdir data')
    if not os.path.isdir('data/'+tgt):
        os.system('mkdir data/'+tgt)
    date0 = df["collected_from"][row][:10]
    already = glob('/home/innereye/astro/data/thumb/'+date0+'_'+tgt+'*')
    if len(already) > 0:
        msg = 'pictures exist:    '
        for alr in already:
            msg += '\n    ' + alr.split('/')[-1]
        print(msg)
    else:
        log_csv = f'/home/innereye/astro/logs/{tgt}_{date0}.csv'
        if os.path.isfile(log_csv):
            chosen_df = pd.read_csv(log_csv)
            files = list(chosen_df['file'][chosen_df['chosen']])
            download_fits_files(files, destination_folder='data/' + tgt)
        else:
            t_min = [np.floor(Time(df['collected_from'][row]).mjd),
                     np.ceil(Time(df['collected_to'][row]).mjd)]
            args = {'target_name': tgt,
                    't_min': t_min,
                    'obs_collection': "JWST",
                    'calib_level': 3,
                    'dataRights': 'public',
                    'intentType': 'science',
                    'dataproduct_type': "image"}
            table = Observations.query_criteria(**args)
            files = list(table['dataURL'])
            files = [x.split('/')[-1] for x in files]
            download_fits_files(files, destination_folder='data/' + tgt)
            chosen_df = choose_fits(files, folder='data/' + tgt)
            chosen_df.to_csv(log_csv, index=False, sep=',')

            # os.chdir('data/'+tgt)
        # check if pkl was saved locally


        ## make image
        # read the files and for each filter, choose smaller and close to target images

        use = chosen_df['chosen'].to_numpy()
        files = np.asarray(chosen_df['file'])
        todelete = files[~use]
        # for rm in todelete:
        #     os.system('rm '+rm)
        # see if we have both MIRI and NIRCam, choose RGB method accordingly
        files = files[use]
        filt = filt_num(files)
        files = files[np.argsort(filt)]
        filt = np.sort(filt)
        mn = np.zeros((len(files),2), bool)
        for ii in range(len(files)):
            if 'miri' in files[ii]:
                mn[ii,0] = True
            if 'nircam' in files[ii]:
                mn[ii,1] = True
            if mn[ii, :].sum() == 0:
                raise Exception('no miri and no nircam')
            elif mn[ii, :].sum() == 2:
                raise Exception('both miri and nircam')

        method = 'rrgggbb'
        if np.mean(mn[:, 0]) == 1:
            instrument = 'MIRI'
        elif np.mean(mn[:,1]) == 1:
            instrument = 'NIRCam'
        else:
            instrument = 'NIRCam+MIRI'
            method = 'mnn'
        os.chdir('data')# os.chdir('..')
        plotted = []
        # TODO decide if to use 0.5 1 1
        if 'MIRI' in instrument:
            auto_plot(tgt, exp=list(files[mn[:, 0]]), png=tgt+'_MIRI.png', pow=[1, 1, 1], pkl=False, resize=True, method='rrgggbb', plot=False)
            plotted.append(tgt+'_MIRI.png')
        if 'NIRCam' in instrument:
            auto_plot(tgt, exp=list(files[mn[:, 1]]), png=tgt + '_NIRCam.png', pow=[1, 1, 1], pkl=False, resize=True, method='rrgggbb', plot=False)
            plotted.append(tgt + '_NIRCam.png')
        if '+' in instrument:
            auto_plot(tgt, exp=list(files), png=tgt+'_'+instrument+'.png', pow=[1, 1, 1], pkl=True, resize=True, method='mnn', plot=False)
            plotted.append(tgt+'_'+instrument+'.png')
        ##
        for ii in range(len(plotted)):
            img = plt.imread(plotted[ii])[..., :3]
            # edge = np.where(np.mean(np.mean(img, 2), 1))[0][0]
            new_height = 300
            ratio = new_height / img.shape[0]
            imgrs = resize(img, (new_height, int(ratio * img.shape[1])))
            plt.imsave('/home/innereye/astro/data/thumb/'+date0+'_'+plotted[ii], imgrs, cmap='gray')


