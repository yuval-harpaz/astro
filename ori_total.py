from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os
from reproject import reproject_interp
from sklearn import decomposition
path = list_files('/home/innereye/JWST/Ori/',search='*.fits')
from astropy.convolution import Gaussian2DKernel, convolve
# path = list_files('ngc_628', search='*miri*.fits')
mfilt = np.where(['m_i2d.fits' in x for x in path])[0][:-1]
path = [path[ii] for ii in mfilt]
## brief preview
meds = 4
crop = [3800,5000,5600,6800]
# FIXME: pad=2, hole_size=50, then smooth remaining zeros
for ii in range(len(path)):
    if ii == 0:
        hdu0 = fits.open(path[ii])
        img = hdu0[1].data
        if len(crop) > 0:
            img = img[crop[0]:crop[1],crop[2]:crop[3]]
        # img = fill_holes(img, pad=1, hole_size=50)
        hdr0 = hdu0[1].header
        total = np.zeros(img.shape)
        del hdu0
    else:
        hdu = fits.open(path[ii])
        reproj, _ = reproject_interp(hdu[1], hdr0)
        img = reproj
        if len(crop) > 0:
            img = img[crop[0]:crop[1],crop[2]:crop[3]]

    if img.shape[0] == 0:
        raise Exception('bad zero')
    img[img == 0] = np.nan
    med = np.nanmedian(img)
    img[np.isnan(img)] = 0
    img = fill_holes(img, pad=1, hole_size=50, if_above=med*meds)
    img = img - (med / meds)
    img = img / (med * meds) * 255
    img[img > 255] = 255
    img[img < 0] = 0
    plt.subplot(2,3,ii+1)
    plt.imshow(img, cmap='hot')
    if ii == 0:
        r = img
    elif ii == 5:
        b = img
    total += img
total = total / len(mfilt)
plt.show(block=False)

with open('ORIBAR.pkl', 'wb') as f:
    pickle.dump([r,total,b], f)

# layers = mosaic(path,method='layers')
rgbt = np.zeros((total.shape[0], total.shape[1], 3))
rgbt[..., 0] = b
rgbt[..., 1] = total
rgbt[..., 2] = r


plt.figure()
plt.imshow(rgbt.astype('uint8'), origin='lower')
plt.show(block=False)

