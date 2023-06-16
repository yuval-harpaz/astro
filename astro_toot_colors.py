import os

from astro_utils import *
from astropy.time import Time
from mastodon_bot import connect_bot
from astro_list_ngc import list_ngc
from glob import glob
from skimage.transform import resize
from pyongc import ongc
##
df = list_ngc()
df_prev = pd.read_csv('ngc.csv', sep=',')
masto, loc = connect_bot()
##
if (df.iloc[0]['collected_to'] <= df_prev.iloc[0]['collected_to']) and (loc == 'github'):
    print('no new NGC')
else:
    if loc == 'local':
        row = 21
        pkl = True
    else:
        row = 0
        pkl = False
    tgt = df['target_name'][row]
    t_min = [np.floor(Time(df_prev['collected_from'][row]).mjd),
             np.ceil(Time(df_prev['collected_to'][row]).mjd)]
    args = {'target_name': tgt,
            't_min': t_min,
            'obs_collection': "JWST",
            'calib_level': 3,
            'dataRights': 'public',
            'intentType': 'science',
            'dataproduct_type': "image"}
    table = Observations.query_criteria(**args)
    if not os.path.isdir('data'):
        os.system('mkdir data')
    if not os.path.isdir('data/'+tgt):
        os.system('mkdir data/'+tgt)
    os.chdir('data/'+tgt)
    # check if pkl was saved locally
    if len(glob('*.pkl')) == 0:
        for url in table['dataURL']:
            # check if data was saved locally
            if not os.path.isfile(url.replace('mast:JWST/product/', '')):
                os.system('wget https://mast.stsci.edu/portal/Download/file/' + url[5:])
    ##
    files = glob('*.fits')
    filt = filt_num(files)
    if len(np.unique(filt)) < len(files):
        raise Exception('duplicate filters')

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
    if np.mean(mn[:,0]) == 1:
        instrument = 'MIRI'
    elif np.mean(mn[:,1]) == 1:
        instrument = 'NIRCam'
    else:
        instrument = 'NIRCam + MIRI'
        method = 'mnn'
    os.chdir('..')
    # TODO: do something about duplicate filters, avoid error when one layer is not in overlap, decide if to use 0.5 1 1
    auto_plot(tgt, '*_i2d.fits', png=True, pow=[1, 1, 1], pkl=pkl, resize=True, method=method, plot=False)
    ##
    img = plt.imread(tgt+'.png')
    size = os.path.getsize(tgt+'.png')
    mb2 = 2 * 10 ** 6  # mastodon allows 2MB
    if size >= mb2:
        ratio = mb2 / size - 0.01
        img = resize(img, (int(ratio ** 0.5 * img.shape[0]), int(ratio ** 0.5 * img.shape[1])))
        plt.imsave(tgt+'.png', img, cmap='gray')

    metadata = masto.media_post(tgt+'.png', "image/PNG")


    what = '.'
    if df['NGC'][row] > 0:
        obj = ongc.get('NGC' + str(df['NGC'][row]))
        if obj is not None:
            what = ', A '+obj.type.lower()+' in '+obj.constellation + '.'
    toot = tgt + what + 'An image of #JWST data released on ' + df_prev['collected_from'][row][:10] +\
           ', instrument: ' + instrument + ', filters: ' + str(filt.astype(int)) + '.\n' + \
        ' For the latest NGC images by JWST, see:\n https://yuval-harpaz.github.io/astro/ngc_grid.html'
    masto.status_post(toot, media_ids=metadata["id"])
    print('toot image')
# search images by observation date and by release date

# df.to_csv('ngc.csv', sep=',', index=False)
# if df.iloc[0]['release_date'] > df.iloc[0]['release_date']:
#     toot = 'A new NGC image at https://yuval-harpaz.github.io/astro/ngc.html'
#     masto, _ = connect_bot()
#     masto.status_post(toot)

##

