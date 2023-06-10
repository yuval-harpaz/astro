import matplotlib.pyplot as plt

from astro_utils import *
from astro_fill_holes import *
# from scipy.ndimage.filters import maximum_filter



##

##
os.chdir('/home/innereye/JWST/SPT0418-47')
layers = np.load('SPT0418-47.pkl', allow_pickle=True)
# layers = layers[1200:1600,1050:1450,:]
for lay in range(layers.shape[2]):
    # layers[:, :, lay] = level_adjust(layers[:, :, lay])
    layers[:, :, lay] = layers[:, :, lay] - np.min(layers[:, :, lay])
    layers[:, :, lay] = layers[:, :, lay] / np.max(layers[:, :, lay])
xy = optimize_xy_manual(layers)
layers = np.load('SPT0418-47.pkl', allow_pickle=True)
shifted = roll(layers, xy, nan_edge=True)
with open('ngc3132.pkl', 'wb') as f:
    pickle.dump(shifted, f)

auto_plot('SPT0418-47', png=True, pow=[1, 1, 1], pkl=True, resize=False, method='mnn')
pow = [2, 2, 2]
rgb = auto_plot('SPT0418-47', png=True, pow=pow, pkl=True, resize=False, method='rrgggbb')
mtn = auto_plot('SPT0418-47', png=True, pow=pow, pkl=True, resize=False, method='mtn')
mix = np.round((mtn.astype(float)+rgb.astype(float))/2)
mix = mix.astype('uint8')
plt.imshow(mix, origin='lower')
plt.imsave('mix.png', mix, origin='lower')
# ##
# # layers = layers[3000:3600,2700:3300, :]
# os.chdir('/home/innereye/JWST/ngc3132')
# layers = np.load('ngc3132.pkl', allow_pickle=True)
# for lay in range(layers.shape[2]):
#     xy = hole_xy(layers[:,:,lay])
#     size = hole_size(layers[:,:,lay], xy, plot=False)
#     layers[:, :, lay] = hole_disk_fill(layers[:,:,lay], xy, size, larger_than=3)
#     print(lay)
#
# with open('ngc3132.pkl', 'wb') as f:
#     pickle.dump(layers, f)
#
# ##
# path = np.asarray(list_files('/home/innereye/JWST/ngc3132/', '*.fits'))
# filt = filt_num(path)
# order = np.argsort(filt)
# filt = filt[order]
# path = path[order]
# auto_plot('ngc3132', exp=path[[0, 1, 5, 6, 8, 9]], png='6layers.png', pow=[0.5, 0.5, 0.5], pkl=True, resize=True, method='rrgggbb')
# auto_plot('ngc3132', png='rrgggbb.png', pow=[0.5, 0.5, 0.5], pkl=True, resize=True, method='rrgggbb')
# auto_plot('ngc3132', png='mnn.png', pow=[0.5, 0.5, 0.5], pkl=True, resize=True, method='mnn', plot=False)
# auto_plot('ngc3132', png='mnn05r.png', pow=[0.5, 1, 1], pkl=True, resize=True, method='mnn', plot=False)
# auto_plot('ngc3132', exp=path[[1, 2, 3,4, 9]], png='5layers.png', pow=[0.5, 0.5, 0.5], pkl=True, resize=True, method='mnn')
# ##
# os.chdir('/home/innereye/JWST/ngc3132')
# layers = np.load('ngc3132.pkl', allow_pickle=True)
# for lay in range(layers.shape[2]):
#     layers[:, :, lay] = level_adjust(layers[:, :, lay], factor=4)
# plt.figure()
# for sp in range(10):
#     plt.subplot(2, 5, sp+1)
#     plt.imshow(layers[:,:,sp]**0.5, cmap='gray')
#     plt.axis('off')
#
# ##
#
#
# img = layers[:, :, 3]
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(level_adjust(img))
# xy = hole_xy(img)
# size = hole_size(img, xy, plot=False)
# img = hole_disk_fill(img, xy, size, larger_than=3)
# plt.subplot(1, 2, 2)
# plt.imshow(level_adjust(img))
#
#
# # for lay in range(layers.shape[2]):
# #     layers[:, :, lay] = level_adjust(layers[:, :, lay], factor=4)
#
#
#
#
#
#
#
#
# bestx, besty, layers = optimize_xy(layers, square_size=[1000, 1000])
#
#
# path = np.asarray(list_files('/home/innereye/JWST/ngc3132/', '*.fits'))
# filt = filt_num(path)
# order = np.argsort(filt)
# filt = filt[order]
# path = path[order]
# auto_plot('ngc3132', exp=path[[0, 1, 5, 6, 8, 9]], png=False, pow=[0.5, 0.5, 0.5], pkl=True, resize=False, method='rrgggbb')
#
#
# for ii in range(layers.shape[2]):
#     plt.subplot(2,5,ii+1)
#     plt.imshow(layers[:,:,ii], origin='lower', cmap='hot')
#     plt.title(path[ii][20:])
#     plt.axis('off')
#
#
# # path = ['./jw02733-o001_t001_nircam_clear-f090w_i2d.fits',
# #        './jw02733-o001_t001_nircam_clear-f187n_i2d.fits',
# #        './jw02733-o001_t001_nircam_clear-f212n_i2d.fits',
# #        './jw02733-o001_t001_nircam_clear-f356w_i2d.fits',
# #        './jw02733-o001_t001_nircam_f405n-f444w_i2d.fits',
# #        './jw02733-o001_t001_nircam_f444w-f470n_i2d.fits',
# #        './jw02733-o002_t001_miri_f770w_i2d.fits',
# #        './jw02733-o002_t001_miri_f1130w_i2d.fits',
# #        './jw02733-o002_t001_miri_f1280w_i2d.fits',
# #        './jw02733-o002_t001_miri_f1800w_i2d.fits']
#
#
#
#
# # rut = '/home/innereye/JWST/SDSSJ1723+3411/MAST_2022-08-31T1707/JWST/'
# # # path = list_files(rut, '*nircam*_i2d.fits')
# # path = list_files(rut, '*_i2d.fits')
# # filter = filt_num(path)
# # order = np.argsort(filter)
# # filter = filter[order]
# # path = path[order]
# # layers = reproject(path, project_to=0)
# #
# # layers = layers[165:, 385:-35, :]
# # for lay in range(len(path)):
# #     layer = layers[:,:,lay]
# #     mask = np.isnan(layer)
# #     layer[mask] = 0
# #     layer = level_adjust(layer)
# #     layer[mask] = np.nan
# #     layers[:, :, lay] = layer
# # #
# # # with open('layers.pkl', 'wb') as f: pickle.dump(layers, f)
# # with open('layers.pkl','rb') as f: layers = pickle.load(f)
# #
# # # col = np.ones((layers.shape[2], 3))
# # # col[:,0] = 94.5/100
# # # col[:,2] = 5.9/100
# # # col[:,1] = filter/np.max(filter)
# #
# # col = matplotlib.cm.jet(filter/np.max(filter))[:, :3]
# # # for ifilt, filt in enumerate(filter):
# # #     col[ifilt] = filt
# #
# # # mean = np.zeros((layers.shape[0], layers.shape[1], 3))
# # # mean[:,:,0] = np.nanmean(layers, 2)*col[0,0]
# # # mean[:,:,2] = np.nanmean(layers, 2)*col[0,2]
# # # ic = 1
# # # for lay in range(layers.shape[2]):
# # #     mean[:, :, ic] = np.nansum(np.array([mean[:, :, ic], layers[:, :, lay] * (1 / layers.shape[2]) * col[lay, ic]]), 0)
# #
# # mean = np.zeros((layers.shape[0], layers.shape[1], 3))
# # for lay in range(layers.shape[2]):
# #     for ic in range(3):
# #         mean[:, :, ic] = np.nansum(np.array([mean[:, :, ic], layers[:, :, lay]*(1/layers.shape[2])*col[lay, ic]]), 0)
# #
# # plt.imshow(mean)
# #
