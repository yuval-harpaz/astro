from astro_utils import *
_ = auto_plot('ring', exp='*.fits', png='factor1.png', pow=[2, 1, 1], pkl=False, method='rrgggbb', resize=False, plot=True, adj_args={'ignore0': True, 'factor': 1}, max_color=True)
##
path = list_files('/media/innereye/My Passport/Data/JWST/data/ring/')
layers = mosaic(path, method='layers', plot=False)
##
lim = [[0, 1],[-0.05, 0],[0,1], [0, 1],[-0.05, 0],[0,1]]
plt.figure()
for ii in range(6):
    plt.subplot(2,3,ii+1)
    plt.imshow(layers[..., ii])
    plt.title(f'{np.round(np.percentile(layers[..., ii], 10),2)} {np.round(np.percentile(layers[..., ii], 90))}')
    plt.clim(lim[ii])
##
ladj = layers.copy()
for ii in range(6):
    ladj[..., ii] = level_adjust(layers[...,ii].copy(), ignore0=True, factor=2)
ladj[ladj == 0] = np.nan
rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
# layers[layers <= 0] = np.nan
rgb[..., 0] = np.nanmean(ladj[..., [1, 4]], axis=2)**2
rgb[..., 1] = np.nanmean(ladj[..., [0, 3]], axis=2)
rgb[..., 2] = np.nanmean(ladj[..., [2, 5]], axis=2)
#
# for ii in range(3):
#     rgb[..., ii] = level_adjust(rgb[..., ii])
plt.figure()
plt.imshow(rgb)
##
plt.imsave('mosaic.png', rgb, origin='lower')