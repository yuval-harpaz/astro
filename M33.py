from astro_utils import *
os.chdir('/media/innereye/My Passport/Data/JWST/M33')
path = ['jw02130-o014_t008_nircam_clear-f430m_i2d.fits', 'jw02128-o001_t001_miri_f560w_i2d.fits',
        '/media/innereye/My Passport/Data/JWST/data/NGC0598MIRI-BRIGHT1/jw02130-o012_t010_miri_f1000w-sub256_i2d.fits',
        '/media/innereye/My Passport/Data/JWST/data/NGC0598MIRI-BRIGHT1/jw02130-o012_t010_miri_f2100w-sub256_i2d.fits']

layers = reproject(path, 2)
for lay in range(layers.shape[2]):
    layers[..., lay] = level_adjust(layers[..., lay])

order = [[3], [1, 2], [0]]
rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
for ic in range(3):
    rgb[..., ic] = np.mean(layers[..., order[ic]], 2)
# rgb = roll(rgb, np.array([[0,0],[0,0],[10,10]]))
# rgb[..., 2] = np.roll(rgb[..., 2], [10,10])
rgb[..., 2] = np.roll(rgb[..., 2], -1, axis=1)
rgb[..., 2] = np.roll(rgb[..., 2], -2, axis=0)
plt.figure()
plt.imshow(rgb, origin='lower')
plt.imsave('bright1.png', rgb, origin='lower')

## bright2
path = ['jw02130-o014_t008_nircam_clear-f430m_i2d.fits', 'jw02128-o001_t001_miri_f560w_i2d.fits',
        '/media/innereye/My Passport/Data/JWST/data/NGC0598MIRI-BRIGHT2/jw02130-o013_t011_miri_f1000w-sub256_i2d.fits',
        '/media/innereye/My Passport/Data/JWST/data/NGC0598MIRI-BRIGHT2/jw02130-o013_t011_miri_f2100w-sub256_i2d.fits']

layers = reproject(path, 2)
for lay in range(layers.shape[2]):
    layers[..., lay] = level_adjust(layers[..., lay])

order = [[3], [1, 2], [0]]
##
rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
for ic in range(3):
    rgb[..., ic] = np.mean(layers[..., order[ic]], 2)
# rgb = roll(rgb, np.array([[0,0],[0,0],[10,10]]))
# rgb[..., 2] = np.roll(rgb[..., 2], [10,10])
rgb[..., 2] = np.roll(rgb[..., 2], -2, axis=1)
rgb[..., 2] = np.roll(rgb[..., 2], 1, axis=0)
plt.figure()
plt.imshow(rgb, origin='lower')
##
plt.imsave('bright2.png', rgb, origin='lower')