from astro_utils import *
from astropy.time import Time
from mastodon_bot import connect_bot
from astro_list_ngc import list_ngc, ngc_html
from glob import glob
from skimage.transform import resize
from pyongc import ongc
##
df_prev = pd.read_csv('ngc.csv', sep=',')
df = list_ngc()
df.to_csv('ngc.csv', sep=',', index=False)
ngc_html()
masto, loc = connect_bot()
##
if (df.iloc[0]['collected_to'] <= df_prev.iloc[0]['collected_to']) and (loc == 'github'):
    print('no new NGC')
else:
    if loc == 'local':
        row = 10
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
    offset = np.zeros(len(files))
    size = np.zeros(len(files))
    for ifile in range(len(files)):
        hdu = fits.open(files[ifile])
        offset[ifile] = np.max(np.abs([hdu[0].header['XOFFSET'], hdu[0].header['YOFFSET']]))
        size[ifile] = hdu[0].header['SUBSIZE1'] * hdu[0].header['SUBSIZE2']
        hdu.close()
    order = np.argsort(size)
    files = np.asarray(files)[order]
    offset = offset[order]
    filt = filt_num(files)
    filtu = np.unique(filt)
    use = np.ones(len(files), bool)
    if len(filtu) < len(files):
        for ii in range(len(filtu)):
            idx = np.where(filt == filtu[ii])[0]
            selected = np.argmin(offset[idx])
            for jj in range(len(idx)):
                if jj != selected:
                    use[idx[jj]] = False
    # sp = 0
    # for ii in range(16):
    #     sp += 1
    #     plt.subplot(4, 4, sp)
    #     hdu = fits.open(files[ii])
    #     plt.imshow(level_adjust(hdu[1].data))
    #     plt.axis('off')
    #     plt.title(str(filt[ii]) + ' ' + str(use[ii]))
    # plt.show(block=False)
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
    if np.mean(mn[:,0]) == 1:
        instrument = 'MIRI'
    elif np.mean(mn[:,1]) == 1:
        instrument = 'NIRCam'
    else:
        instrument = 'NIRCam + MIRI'
        method = 'mnn'
    os.chdir('..')
    # TODO: do something about duplicate filters, avoid error when one layer is not in overlap, decide if to use 0.5 1 1
    auto_plot(tgt, exp=list(files), png=True, pow=[1, 1, 1], pkl=pkl, resize=True, method=method, plot=False)
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
    toot = tgt + what + ' An image of #JWST data released on ' + df_prev['collected_from'][row][:10] +\
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

