import pandas as pd

from astro_list_ngc import make_thumb
from astro_utils import *

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
# from astro_utils import *
# from glob import glob
# os.chdir('/home/innereye/JWST/neptune/')
# files = np.sort(glob('jw*s3d.fits'))
# files1d = np.sort(glob('jw*x1d.fits'))
#
# data = []
# for ii in [0, 1, 2, 3]:
#     hdu = fits.open(files[ii])
#     data.append(hdu[1].data.copy())
#     hdu.close()
# wavelength = []
# for ii in [0, 1, 2, 3]:
#     hdu = fits.open(files1d[ii])
#     wavelength.append(hdu[1].data['WAVELENGTH'])
#     hdu.close()
#
# plt.figure()
# for ii in [0, 1, 2, 3]:
#     plt.subplot(2,2,ii+1)
#     med = np.nanmedian(data[ii], 0)
#     if ii == 0:
#         med[med < 0] = 0
#         med[med > 200] = 0
#     plt.imshow(med)
#
# center = [[22, 21], [20, 23], [25, 24], [12, 16]]
# corner = [[13, 17], [15, 19], [21, 20], [10, 16]]
# buldge = [[25, 25], [22, 24], [27, 25], [13, 18]]
# plt.figure()
# for ii in [0, 1, 2, 3]:
#     plt.subplot(2,2,ii+1)
#     med = np.nanmedian(data[ii], 0)
#     if ii == 0:
#         med[med < 0] = 0
#         med[med > 200] = 0
#     # med[center[ii][0], center[ii][1]] = 0
#     med[buldge[ii][0], buldge[ii][1]] = 0
#     plt.imshow(med)
#
# plt.figure()
# for ii in [0, 1, 2, 3]:
#     # plt.subplot(2,2,ii+1)
#     plt.plot(wavelength[ii], data[ii][:, corner[ii][0], corner[ii][1]], 'g')
#     plt.plot(wavelength[ii], data[ii][:, buldge[ii][0], buldge[ii][1]], 'b')
#     plt.plot(wavelength[ii], data[ii][:, center[ii][0], center[ii][1]], 'r')
# plt.title('Flux for Neptune at three spots')
# plt.ylabel(hdu[1].header['TUNIT2'])
# plt.xlabel('Wavelength (µm)')
# plt.grid()
#
#
#
