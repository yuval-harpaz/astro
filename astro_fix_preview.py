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
exp = ['jw01328-o030_t010_miri_f560w-sub128_i2d.fits', 'jw01328-o030_t010_miri_f770w-sub128_i2d.fits',
       'jw01328-o030_t010_miri_f1500w-sub128_i2d.fits']
auto_plot('NGC-7469', exp=exp, png='miri.png', pow=[1, 1, 1], pkl=False, resize=True, method='rrgggbb', plot=True)
#
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
