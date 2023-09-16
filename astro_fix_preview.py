import pandas as pd

from astro_list_ngc import make_thumb, ngc_html_thumb
from astro_utils import *
from glob import glob
auto_plot('NGC3256-CENTERED', exp='*o029*.fits', png='NGC3256-CENTERED_MIRI.png', pow=[1, 1, 1], pkl=False, resize=True, method='rrgggbb', plot=False)
auto_plot('NGC3256-CENTERED', exp='log', png='NGC3256-CENTERED_NIRCam+MIRI.png', pow=[1, 1, 1], pkl=True, resize=True, method='mnn', plot=False)
make_thumb(glob('*MIRI*.png'), '2022-12-25')

## big mess. make an image of the two large scans
auto_plot('NGC-7469', exp='logNGC-7469_2022-07-01.csv', png='test.png', pow=[0.75, 1, 1], pkl=False, resize=True, method='mnn', plot=True)

make_thumb('NGC-7469_NIRCam+MIRI.png', '2022-07-01')
exp = ['jw01328-o019_t010_nircam_clear-f335m_i2d.fits','jw01328-o019_t010_nircam_clear-f150w_i2d.fits','jw01328-o019_t010_nircam_clear-f200w_i2d.fits','jw01328-o019_t010_nircam_clear-f444w_i2d.fits']
auto_plot('NGC-7469', exp=exp, png='nircam.png', pow=[1, 1, 1], pkl=False, resize=True, method='rrgggbb', plot=True)
exp = ['jw01328-o030_t010_miri_f560w-sub128_i2d.fits', 'jw01328-o030_t010_miri_f770w-sub128_i2d.fits', 'jw01328-o030_t010_miri_f1500w-sub128_i2d.fits']
auto_plot('NGC-7469', exp=exp, png='miri.png', pow=[1, 1, 1], pkl=False, resize=True, method='rrgggbb', plot=True)

logs = glob('/home/innereye/astro/logs/*7469*')
df = pd.read_csv(logs[0])
for ii in [1,2,3]:
    df = pd.concat([df, pd.read_csv(logs[ii])])
files = list(df['file'][df['width'] > 1000])
#
plt.figure()
for ii in range(7):
    plt.subplot(2,4,ii+1)
    try:
       hdu = fits.open('/media/innereye/My Passport/Data/JWST/data/NGC-7469/' + files[ii])
    except:
           hdu = fits.open('/media/innereye/My Passport/Data/JWST/data/NGC-7469-MRS/' + files[ii])
    plt.imshow(level_adjust(hdu[1].data))
    plt.title(files[ii])
#
files = list(df['file'][df['file'].str.contains('brightsky')])[:3] + \
        list(df['file'][df['width'] > 2000])
auto_plot('NGC-7469', exp=files, png='large.png', pow=[1, 1, 1], pkl=False, resize=True, method='mnn', plot=True)

##
make_thumb('NGC-7469-MRS', '2022-07-04')
##

logs = glob('/home/innereye/astro/logs/*3324*')
df = pd.read_csv(logs[0])
df = pd.concat([df, pd.read_csv(logs[1])])
auto_plot('NGC-3324', png='both.png', pow=[1, 1, 1], pkl=False, resize=True, method='mnn', plot=True)
##
auto_plot('NGC-3132', png='both.png', pow=[1, 1, 1], pkl=False, resize=True, method='mnn', plot=True)
make_thumb('NGC-3132_NIRCam+MIRI.png', '2022-06-03')
ngc_html_thumb()
##
df = pd.read_csv('logs/NGC-4321_2023-01-17.csv')
# files = list(df['file'][df['offset'] < 100])
files = list(df['file'])
plt.figure()
for ii in range(16):
    plt.subplot(4,4,ii+1)
    hdu = fits.open('/media/innereye/My Passport/Data/JWST/data/NGC-4321/' + files[ii])
    plt.imshow(level_adjust(hdu[1].data))
    plt.title(files[ii])
os.chdir('/media/innereye/My Passport/Data/JWST/data/NGC-4321/')
plotted = glob('*png')
make_thumb(plotted, '2023-01-17', flip=False)
##
'''
sep 2023 issue with two color images
'''
##
png = 'LDN-694_NIRCam.png'
auto_plot('LDN-694', exp='log', png=png, pkl=True, resize=True, method='rrgggbb', plot=False,
          max_color=False, fill=False, deband=False, adj_args={'factor': 3})
date = glob(f'/home/innereye/astro/docs/thumb/*{png}')[0].split('/')[-1][:10]
make_thumb(png, date, flip=False)
##
tgt = 'CL-WESTERLUND-CF'
png = tgt + '_NIRCam.png'
auto_plot(tgt, exp='*nircam*fits', png=png, pow=[1, 1, 1], pkl=False,
          resize=True, method='rrgggbb', plot=False)
date = glob(f'/home/innereye/astro/docs/thumb/*{png}')[0].split('/')[-1][:10]
make_thumb(png, date, flip=False)
##
tgt = 'NGC-6822-NIRCAM-TILE-1'
inst = 'MIRI'
png = tgt + f'_{inst}.png'
auto_plot(tgt, exp=f'*{inst.lower()}*fits', png=png, pow=[1, 1, 1], pkl=False,
          resize=True, method='rrgggbb', plot=False)
date = glob(f'/home/innereye/astro/docs/thumb/*{png}')[0].split('/')[-1][:10]
make_thumb(png, date, flip=False)
##
tgt = 'PREIMAGING+BRICK13'
inst = 'NIRCam'
png = tgt + f'_{inst}.png'
auto_plot(tgt, exp=f'*{inst.lower()}*fits', png=png, pow=[1, 1, 1], pkl=False,
          resize=True, method='rrgggbb', plot=False)
date = glob(f'/home/innereye/astro/docs/thumb/*{png}')[0].split('/')[-1][:10]
make_thumb(png, date, flip=False)
##
tgt = 'M31-NIRCAM-PREIMAGING'
inst = 'NIRCam'
png = tgt + f'_{inst}.png'
auto_plot(tgt, exp=f'*{inst.lower()}*fits', png=png, pow=[1, 1, 1], pkl=False,
          resize=True, method='rrgggbb', plot=False)
date = glob(f'/home/innereye/astro/docs/thumb/*{png}')[0].split('/')[-1][:10]
make_thumb(png, date, flip=False)
##
tgts = ['M-33', 'NGC0300MIRI', 'NGC891-DISK-NORTH3', 'NGC891-DISK-NORTH1', 'NGC891-DISK-NORTH', 'CASSIOPEIA-A-CENTER-IFU',
        'NGC0598MIRI-BRIGHT2', 'NGC0598MIRI-BRIGHT1', 'NGC0598MIRI', 'SGRA', 'NGC7793MIRI', 'NGC2506G31', 'NGC-5139',
        'NGC-6543']
insts = ['MIRI', 'MIRI', 'NIRCam', 'NIRCam', 'NIRCam', 'MIRI',
         'MIRI', 'MIRI', 'MIRI', 'NIRCam', 'MIRI', 'NIRCam', 'NIRCam',
         'MIRI']
for ii in range(len(tgts)):
    tgt = tgts[ii]
    inst = insts[ii]
    png = tgt + f'_{inst}.png'
    auto_plot(tgt, exp=f'*{inst.lower()}*fits', png=png, pow=[1, 1, 1], pkl=False,
              resize=True, method='rrgggbb', plot=False)
    date = glob(f'/home/innereye/astro/docs/thumb/*{png}')[0].split('/')[-1][:10]
    make_thumb(png, date, flip=False)
##
ngc_html_thumb()
##
# from astro_utils import *
