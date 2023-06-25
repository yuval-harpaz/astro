from astro_list_ngc import remake_thumb
remake_thumb()
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
