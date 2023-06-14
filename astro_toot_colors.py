import os

from astro_utils import *
from astropy.time import Time, TimeMJD
from mastodon_bot import connect_bot
from astro_list_ngc import list_ngc
from glob import glob
from skimage.transform import resize
##
df = list_ngc()
df_prev = pd.read_csv('ngc.csv', sep=',')

##
ii = 34
tgt = df_prev['target_name'][ii]
t_min = [np.floor(Time(df_prev['collected_from'][ii]).mjd),
         np.ceil(Time(df_prev['collected_to'][ii]).mjd)]
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
for url in table['dataURL']:
    os.system('wget https://mast.stsci.edu/portal/Download/file/' + url[5:])
##
files = glob('*.fits')
mn = np.zeros((len(files),2), bool)
for ii in range(len(files)):
    if 'miri' in files[ii]:
        mn[ii,0] = True
    if 'nircam' in files[ii]:
        mn[ii,0] = True
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
auto_plot(tgt, '*_i2d.fits', png=True, pow=[1, 1, 1], pkl=False, resize=True, method=method, plot=False)
##
img = plt.imread(tgt+'.png')
size = os.path.getsize(tgt+'.png')
mb2 = 2 * 10 ** 6  # mastodon allows 2MB
if size >= mb2:
    ratio = mb2 / size
    img = resize(img, (int(ratio ** 0.5 * img.shape[0]), int(ratio ** 0.5 * img.shape[1])))
    plt.imsave(tgt+'.png', img, cmap='gray')
masto, _ = connect_bot()
metadata = masto.media_post(tgt+'.png', "image/PNG")
masto.status_post(tgt+' WORK IN PROGRESS, testing script', media_ids=metadata["id"])
print('toot image')
# search images by observation date and by release date

# df.to_csv('ngc.csv', sep=',', index=False)
# if df.iloc[0]['release_date'] > df.iloc[0]['release_date']:
#     toot = 'A new NGC image at https://yuval-harpaz.github.io/astro/ngc.html'
#     masto, _ = connect_bot()
#     masto.status_post(toot)

##

