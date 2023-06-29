# import pandas as pd
# import os.path
import os

import pandas as pd

from astro_utils import *
from astropy.time import Time
# from mastodon_bot import connect_bot
from astro_list_ngc import choose_fits, make_thumb
from glob import glob

# from pyongc import ongc
##
df = pd.read_csv('ngc.csv', sep=',')
# df = list_ngc()
# df.to_csv('ngc.csv', sep=',', index=False)
# ngc_html()
# masto, loc = connect_bot()
##
# loc == 'local'
logs = np.asarray(glob('logs/*.csv'))
okay = np.zeros(len(logs))
for ii in range(len(logs)):
    log = pd.read_csv(logs[ii])
    tgt = logs[ii].split('/')[1].split('_')[0]
    date0 = logs[ii][-14:-4]
    drive = '/media/innereye/My Passport/Data/JWST/'
    if os.path.isdir(drive):
        os.chdir(drive)
    else:
        raise Exception('where is the drive?')
        # os.system('mkdir data')
    if not os.path.isdir('data/'+tgt):
        raise Exception('no data for '+tgt)
    already = glob('/home/innereye/astro/docs/thumb/'+date0+'_'+tgt+'*_large.png')
    if len(already) > 0:
        msg = 'pictures exist:    '
        for alr in already:
            msg += '\n    ' + alr.split('/')[-1]
        print(msg)
    elif tgt in ['NGC2506G31', 'NGC-5139']:  # clusters
        print('no clusters for now '+date0+' '+tgt)
    elif tgt == 'ORIBAR-IMAGING-MIRI':
        print(tgt + ' too messy, two sessions with strange overlap + NIRCam')
    elif 'background' in tgt.lower() or 'BKG' in tgt:
        print('no background for now '+tgt)
    else:
        files = np.asarray(list(log['file'][log['chosen']]))
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
        made_png = False
        if np.sum(mn[:,0]) > 2:
            auto_plot(tgt, exp=list(files[mn[:, 0]]), png=tgt+'_MIRI_large.png',
                      pow=[1, 1, 1], pkl=False, resize=False, method='rrgggbb', plot=False)
            plotted.append(tgt+'_MIRI.png')
            made_png = True
        if np.sum(mn[:,1]) > 2:
            auto_plot(tgt, exp=list(files[mn[:, 1]]), png=tgt + '_NIRCam_large.png', pow=[1, 1, 1], pkl=False, resize=False, method='rrgggbb', plot=False)
            plotted.append(tgt + '_NIRCam_large.png')
            made_png = True
        if '+' in instrument and np.sum(mn) > 2:
            if os.path.isfile(tgt+'.pkl'):
                os.system(f'mv {tgt}.pkl {tgt}_rs.pkl')
            auto_plot(tgt, exp=list(files), png=tgt+'_'+instrument+'.png', pow=[1, 1, 1], pkl=True, resize=False, method='mnn', plot=False)
            plotted.append(tgt+'_'+instrument+'_large.png')
            made_png = True
        ##
        if made_png:
            make_thumb(plotted, date0)
            print('DONE ' + date0 + '_' + tgt)
        else:
            print('no plots for '+ date0 + '_' + tgt)
        # for ii in range(len(plotted)):
        #     img = plt.imread(plotted[ii])[..., :3]
        #     # edge = np.where(np.mean(np.mean(img, 2), 1))[0][0]
        #     new_height = 300
        #     ratio = new_height / img.shape[0]
        #     imgrs = resize(img, (new_height, int(ratio * img.shape[1])))
        #     plt.imsave('/home/innereye/astro/docs/thumb/'+date0+'_'+plotted[ii], imgrs, cmap='gray')



