from astro_utils import *
os.chdir('/media/innereye/My Passport/Data/JWST/MWC-758/')

auto_plot('MWC-758', exp='*.fits', png='tmp2.jpg', pkl=True, resize=False, method='rrgggbb',
          plot=False, fill=True, deband=False, adj_args={'factor': 2}, blc=False, annotate=True)

layers = np.load('MWC-758.pkl', allow_pickle=True)
layers8 = np.zeros((211, 207, 8))
for ii in range(8):
    layers8[..., ii] = np.nanmin([level_adjust(layers[..., ii*2], factor=2),
                                  level_adjust(layers[..., ii*2+1], factor=2)], 0)

rgb = np.zeros((211, 207, 3))
rgb[..., 0] = np.nanmean(layers8[..., [6,7]], 2)
rgb[..., 1] = np.nanmean(layers8[..., [2, 3, 4, 5]], 2)
rgb[..., 2] = np.nanmean(layers8[..., [0, 1]], 2)
plt.imshow(rgb, origin='lower')

rgb = np.zeros((211, 207, 3))
rgb[..., 0] = np.nanmean(layers8[..., [5, 6, 7]], 2)
rgb[..., 1] = np.nanmean(layers8[..., [3, 4]], 2)
rgb[..., 2] = np.nanmean(layers8[..., [0, 1, 2]], 2)
plt.imshow(rgb, origin='lower')
