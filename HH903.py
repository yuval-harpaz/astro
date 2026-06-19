# import os.path
# import pickle
#
# import matplotlib.pyplot as plt
# from matplotlib import colors
from astro_utils import *
os.chdir('/media/yuval/KINGSTON/JWST/data/HH-903')

files = glob('*.fits')
num = filt_num(files)

layers = np.load('HH-903.pkl', allow_pickle=True)
img = np.zeros((layers.shape[0], layers.shape[1], 3))
img[..., 0] = level_adjust(layers[..., 3], factor=2)-level_adjust(layers[..., 2], factor=2)
img[..., 1] = level_adjust(layers[..., 1], factor=2)-level_adjust(layers[..., 0], factor=2)
img[..., 2] = level_adjust(layers[..., 0], factor=2)

img[img < 0] = 0
plt.imshow(img, origin='lower')

plt.imsave('HH-903.jpg', img, origin='lower', pil_kwargs={'quality':95})
