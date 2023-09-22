from astro_utils import *
from astro_fill_holes import *
# from time import time
os.chdir('/media/innereye/My Passport/Data/JWST/data/GALCEN')
# red = np.load('GALCEN.pkl', allow_pickle=True)[..., -1]
##
hdu = fits.open('jw01939-o001_t001_nircam_f405n-f444w_i2d.fits')
img = hdu[1].data.copy()[1000:2000, 3000:4000]
# img = hole_func_fill(img, func='mean', fill_below=1)
img = hole_func_fill(img)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(hdu[1].data[1000:2000, 3000:4000])
plt.clim(0, 5000)
plt.subplot(1,2,2)
plt.imshow(img)
plt.clim(0, 5000)
##
auto_plot('GALCEN', exp='*.fits', png='fill2.png', pkl=True, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=True, deband=False, adj_args={'factor': 2})
##



# hdu = fits.open('jw01939-o001_t001_nircam_f405n-f444w_i2d.fits')
# funcs = ['nanmean', 'nanmax', 'nanmedian']
# plt.figure()
# sp = 0
# for func in funcs:
#     sp += 1
#     func = eval('np.'+func)
#     img = hdu[1].data.copy()[1000:2000, 3000:4000]
#     kernel = Gaussian2DKernel(x_stddev=2)
#     img = hole_func_fill(img, func=func, kernel=kernel)
#     plt.subplot(2, 2, sp)
#     plt.imshow(img)
#     plt.clim(0, 5000)
# plt.subplot(2, 2, 4)
# plt.imshow(hdu[1].data.copy()[1000:2000, 3000:4000])
# plt.clim(0, 5000)
# ##
# t0 = time()
# filled = hole_conv_fill(red.copy(), n_pixels_around=0, ringsize=25, clean_below_local=0.75, clean_below=0)
# t1 = np.round(time()-t0, 1)  # 348.9
# print('ringsize=25 '+str(t1))
# ##
# #  TODO: move this cell into a function - fill_holes_func, allow max, median ring, gaus etc
# t0 = time()
# img = red.copy()
# img[np.isnan(img)] = 0
# for ii in range(img.shape[0]):
#     pos = np.where(img[ii,:] > 0)[0]
#     if len(pos) == 0:
#         img[ii,:] = np.nan
#     else:
#         img[ii, :pos[0]] = np.nan
#         img[ii, pos[-1]+1:] = np.nan
# for jj in range(img.shape[1]):
#     pos = np.where(img[:, jj] > 0)[0]
#     if len(pos) == 0:
#         img[:, jj] = np.nan
#     else:
#         img[:pos[0], jj] = np.nan
#         img[pos[-1]+1:, jj] = np.nan
# #
# # conv = median_filter(img, footprint=kernel.array)
# # turn nans to zeros for later filling
# kernel = Ring2DKernel(25, 3)
# zer = np.where(img <= 0)
# zer = np.asarray(zer).T
# zer = zer[(zer[:,0] > np.floor(kernel.shape[0]/2)) &
#           (zer[:,1] > np.floor(kernel.shape[0]/2)) &
#           (zer[:,0] < img.shape[0]-np.ceil(kernel.shape[0]/2)) &
#           (zer[:,1] < img.shape[1]-np.ceil(kernel.shape[0]/2)), :]
# img[img == 0] = np.nan  # turn zeros to nans to ignore when computing fill values
# half = int(kernel.shape[0]/2)
# for izer in range(len(zer)):
#     sq = img[zer[izer, 0] - half:zer[izer, 0] + half + 1, zer[izer, 1] - half:zer[izer, 1] + half + 1]
#     img[zer[izer, 0], zer[izer, 1]] = np.nansum(sq*kernel.array)
# img[np.isnan(img)] = 0
# t1 = np.round(time()-t0, 1)  # 348.9
# print('manual 25 '+str(t1))
#
#
# t0 = time()
# conv = convolve(sq, kernel)
# t1 = np.round(time()-t0, 100)  #
# print('conv sq '+str(t1))
# t0 = time()
# man = np.nansum(sq*kernel.array)
# t1 = np.round(time()-t0, 100)  #
# print('manual sq '+str(t1))
# #
# # hdu = fits.open('jw03707-o013_t008_miri_f2100w_i2d.fits')
# # filled = hole_conv_fill(hdu[1].data, n_pixels_around=0, ringsize=5, clean_below_local=0.75, clean_below=0)
# # hdu0 = fits.open('jw03707-o014_t008_nircam_clear-f150w_i2d.fits')
# # reproj, _ = reproject_interp(hdu[1], hdu0[1].header)
#
# # auto_plot('NGC1559', exp='log', png='nircam_deband.png', pkl=True, resize=False, method='rrgggbb', plot=False,
# #           max_color=False, fill=False, deband=2, adj_args={'factor': 3})
# #
# # # os.chdir('/media/innereye/My Passport/Data/JWST/data/HH211NIRCAM/')
# #
# # rgb = auto_plot('NGC1559', exp='log', png='nircam_deband.png', pkl=True, resize=False, method='rrgggbb', plot=False,
# #           max_color=False, fill=False, deband=2, adj_args={'factor': 3})
# #
# # remake = rgb.copy()
# # remake[..., 0] = np.max([rgb[..., 0], np.min(rgb[..., 1:], 2)], 0)
# #
# #
# # ##
# # rgb = auto_plot('NGC1559', exp='log', png='deband_w.png', pkl=True, resize=False, method='mnnw', plot=False,
# #           max_color=False, fill=False, deband=False, adj_args={'factor': 3})
# #
# # ##
# # os.chdir('/media/innereye/My Passport/Data/JWST/data/NGC1559/')
# # # rgb = plt.imread('deband.png')[..., :3]
# # # rgb[..., 0] = np.max([rgb[..., 0], np.min(rgb[..., 1:], 2)], 0)
# # path = glob('*.fits')
# # filt = filt_num(path)
# # order = np.argsort(filt)
# # filt = filt[order]
# # path = np.array(path)[order]
# #
# # layers = np.load('NGC1559.pkl', allow_pickle=True)
# # plt.figure()
# # for lay in range(layers.shape[2]):
# #     layers[..., lay] = level_adjust(layers[..., lay], factor=3)
# #     plt.subplot(2,3, lay+1)
# #     plt.imshow(layers[..., lay], origin='lower')
# #
# # plt.figure()
# # for lay in range(layers.shape[2]):
# #     plt.subplot(2,3, lay+1)
# #     plt.imshow(layers[2800:3600, 2800:3600, lay], origin='lower')
# #
# #
# # rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
# # rgb[..., 0] = layers[..., 3]
# # mean = (layers[..., 2] + layers[..., 1])/2
# # mean[mean > 1] = 1
# # rgb[..., 1] = mean
# # rgb[..., 2] = layers[..., 0]
# # plt.imshow(rgb, origin='lower')
# # rgb[..., 0] = np.max([layers[..., 4], np.min(rgb[..., 1:], 2)], 0)
# # plt.imshow(rgb, origin='lower')
# # rgb[..., 0] = np.max([layers[..., 3], rgb[..., 0]], 0)
# # plt.imshow(rgb, origin='lower')
# # rgb[..., 0] = np.max([layers[..., 5], rgb[..., 0]], 0)
# # plt.figure()
# # plt.imshow(rgb, origin='lower')
