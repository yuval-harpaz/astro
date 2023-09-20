from astro_utils import *
os.chdir('/media/innereye/My Passport/Data/JWST/data/NGC1559')
layers = np.load('NGC1559.pkl', allow_pickle=True)

hdu = fits.open('jw03707-o013_t008_miri_f2100w_i2d.fits')
filled = hole_conv_fill(hdu[1].data, n_pixels_around=0, ringsize=5, clean_below_local=0.75, clean_below=0)
hdu0 = fits.open('jw03707-o014_t008_nircam_clear-f150w_i2d.fits')
reproj, _ = reproject_interp(hdu[1], hdu0[1].header)

# auto_plot('NGC1559', exp='log', png='nircam_deband.png', pkl=True, resize=False, method='rrgggbb', plot=False,
#           max_color=False, fill=False, deband=2, adj_args={'factor': 3})
#
# # os.chdir('/media/innereye/My Passport/Data/JWST/data/HH211NIRCAM/')
#
# rgb = auto_plot('NGC1559', exp='log', png='nircam_deband.png', pkl=True, resize=False, method='rrgggbb', plot=False,
#           max_color=False, fill=False, deband=2, adj_args={'factor': 3})
#
# remake = rgb.copy()
# remake[..., 0] = np.max([rgb[..., 0], np.min(rgb[..., 1:], 2)], 0)
#
#
# ##
# rgb = auto_plot('NGC1559', exp='log', png='deband_w.png', pkl=True, resize=False, method='mnnw', plot=False,
#           max_color=False, fill=False, deband=False, adj_args={'factor': 3})
#
# ##
# os.chdir('/media/innereye/My Passport/Data/JWST/data/NGC1559/')
# # rgb = plt.imread('deband.png')[..., :3]
# # rgb[..., 0] = np.max([rgb[..., 0], np.min(rgb[..., 1:], 2)], 0)
# path = glob('*.fits')
# filt = filt_num(path)
# order = np.argsort(filt)
# filt = filt[order]
# path = np.array(path)[order]
#
# layers = np.load('NGC1559.pkl', allow_pickle=True)
# plt.figure()
# for lay in range(layers.shape[2]):
#     layers[..., lay] = level_adjust(layers[..., lay], factor=3)
#     plt.subplot(2,3, lay+1)
#     plt.imshow(layers[..., lay], origin='lower')
#
# plt.figure()
# for lay in range(layers.shape[2]):
#     plt.subplot(2,3, lay+1)
#     plt.imshow(layers[2800:3600, 2800:3600, lay], origin='lower')
#
#
# rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
# rgb[..., 0] = layers[..., 3]
# mean = (layers[..., 2] + layers[..., 1])/2
# mean[mean > 1] = 1
# rgb[..., 1] = mean
# rgb[..., 2] = layers[..., 0]
# plt.imshow(rgb, origin='lower')
# rgb[..., 0] = np.max([layers[..., 4], np.min(rgb[..., 1:], 2)], 0)
# plt.imshow(rgb, origin='lower')
# rgb[..., 0] = np.max([layers[..., 3], rgb[..., 0]], 0)
# plt.imshow(rgb, origin='lower')
# rgb[..., 0] = np.max([layers[..., 5], rgb[..., 0]], 0)
# plt.figure()
# plt.imshow(rgb, origin='lower')
