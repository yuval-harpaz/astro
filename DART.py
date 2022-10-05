from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os
from reproject import reproject_interp
from sklearn import decomposition
import pickle


path = list_files('/home/innereye/JWST/DART/',search='*n-sub400p_i2d.fits')
plt.figure()
layer = np.zeros((402,400, len(path)))
for ii in range(len(path)):
    hdu = fits.open(path[ii])
    plt.subplot(3,4,ii+1)
    plt.imshow(hdu[1].data, origin='lower', cmap='gray')
    plt.clim(3,50)
    plt.axis('off')
    plt.title(path[ii][10:14])
    layer[:,:,ii] = hdu[1].data
plt.show(block=False)

layer[layer < 0] = 0
avg = np.nanmean(layer, axis=2) ** 0.5

plt.figure()
plt.imshow(avg, cmap='hot', origin='lower')
plt.clim(2, 22)
plt.axis('off')
plt.show(block=False)

avg1 = np.nanmean(layer, axis=2)

plt.figure()
plt.imshow(avg1, cmap='hot', origin='lower')
plt.clim(4, 500)
plt.axis('off')
plt.show(block=False)