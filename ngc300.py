import matplotlib.pyplot as plt

from astro_utils import *
os.chdir('/media/innereye/My Passport/Data/JWST/data/NGC0300MIRI/')
path = glob('*fits')
layers = reproject(path)
for ii in [0, 1]:
    layers[..., ii] = level_adjust(layers[..., ii])


plt.figure()
plt.imshow(layers[..., [1, 0, 0]])

rgb = layers[..., [1, 0, 0]].copy()
for row in range(rgb.shape[0]):
    vec = rgb[row, :, 0].copy()
    vec = vec[vec > 0]
    rgb[row, :, 0] = rgb[row, :, 0] - np.median(vec)
    print(row)

rgb[..., 0] = rgb[..., 0] + np.min(rgb[..., 0])
rgb[..., 0] = level_adjust(rgb[..., 0])
rgb[..., 1] = (rgb[..., 0] + rgb[..., 2]) / 2
plt.figure()
plt.imshow(rgb)

plt.imsave('ngc300.png', rgb, origin='lower')