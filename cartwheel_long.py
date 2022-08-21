from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os
path = list_files('cartwheel', search='*.fits', exclude='clear')
median = mosaic(path, plot=True, method='median')
mn = 0.11
mx = 1.7
plt.clim(mn, mx)
plt.show(block=False)
img = (median-mn)/(mx-mn)*255
img[img < 0] = 0
img[img > 255] = 255
img = img.astype(np.uint8)
img[1947:1952, 1019:1024] = 255
img[775:777, 2042:2045] = 255
plt.imsave('median.png', img.T, cmap='gray')
plt.imsave('median_hot.png', img.T, cmap='hot')
layers = mosaic(path, plot=False, method='layers')
layers[layers == 0] = np.nan
rgb = np.zeros((median.shape[0], median.shape[1], 3))
for ii in range(3):
    layer1 = np.nanmedian(layers[:, :, ii*8:ii*8+8], axis=2)
    layer1[np.isnan(layer1)] = np.nanmedian(layer1)
    rgb[:, :, ii] = layer1
img = (rgb - mn) / (mx - mn) * 255
img[img < 0] = 0
img[img > 255] = 255
img = img.astype(np.uint8)
img[1945:1952, 1018:1025] = 255
img[773:777, 2042:2046] = 255
plt.imsave('median_rgb.png', img.swapaxes(0, 1))
print('images saved here: '+os.getcwd())
print('Buttonpress the figure to kill program')
plt.waitforbuttonpress()

