from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os
from reproject import reproject_interp
from sklearn import decomposition
import pickle


path = list_files('/home/innereye/JWST/vv114/',search='*.fits')[:-1]
filt = filt_num(path)
path = np.asarray(path)[np.flipud(np.argsort(filt))]

# from astropy.convolution import Gaussian2DKernel, convolve
# path = list_files('ngc_628', search='*miri*.fits')
# mfilt = np.where(['m_i2d.fits' in x for x in path])[0][:-1]
# path = [path[ii] for ii in mfilt]
## brief preview

# crop = [3800,5000,5600,6800]


if os.path.isfile('6layers.pkl'):
    layers = np.load('6layers.pkl', allow_pickle=True)
else:
    for ii in range(len(path)):
        if ii == 0:
            hdu0 = fits.open(path[ii])
            img = hdu0[1].data
            # img = img[crop[0]:crop[1],crop[2]:crop[3]]
            # img = fill_holes(img, pad=1, hole_size=50)
            hdr0 = hdu0[1].header
            layers = np.zeros((img.shape[0], img.shape[1], 6))
            del hdu0
        else:
            hdu = fits.open(path[ii])
            reproj, _ = reproject_interp(hdu[1], hdr0)
            img = reproj
            # img = img[crop[0]:crop[1],crop[2]:crop[3]]

        if img.shape[0] == 0:
            raise Exception('bad zero')
        # img[img == 0] = np.nan
        # med = np.nanmedian(img)
        # img[np.isnan(img)] = 0
        layers[:,:,ii] = img
    with open('6layers.pkl', 'wb') as f:
        pickle.dump(layers, f)

meds = 24
total = np.zeros(layers.shape[:2])
norm = np.zeros(layers.shape)
plt.figure()
for ii in range(layers.shape[2]):
    img = layers[:,:,ii]
    # if ii == 0:
    #     img, mask = fill_holes(img, pad=2, hole_size=50, if_above=med*meds, op_mask=True)
    # elif ii > 2:
    #     img = fill_holes(img, pad=2, hole_size=50, if_above=med * meds, ip_mask=mask)
    # else:
    #     img = fill_holes(img, pad=2, hole_size=50, if_above=med * meds)
    img = img**0.5
    img[img == 0] = np.nan
    med = np.nanmedian(img)
    img[np.isnan(img)] = 0
    img = img - (med / meds)
    img = img / (med * meds - med / meds) * 255
    img[img > 255] = 255
    img[img < 0] = 0
    img = fill_holes(img, pad=1, hole_size=20, if_above=150, fill_below=0)
    plt.subplot(2,3,ii+1)
    plt.imshow(img, cmap='hot')
    if ii == 0:
        r = img
    elif ii == 5:
        b = img
    total += img
    norm[:,:,ii] = img
total = total / layers.shape[2]
plt.show(block=False)



# layers = mosaic(path,method='layers')
rgbt = np.zeros((total.shape[0], total.shape[1], 3))
rgbt[..., 0] = b
rgbt[..., 1] = total
rgbt[..., 2] = r
plt.figure()
plt.imshow(rgbt.astype('uint8'), origin='lower')
plt.clim(20,200)
plt.show(block=False)


rgbt = np.zeros((total.shape[0], total.shape[1], 3))
rgbt[..., 0] = np.mean(norm[:, :, :4]*1.5, axis=2)
rgbt[..., 1] = total
rgbt[..., 2] = np.mean(norm[:, :, 2:], axis=2)
plt.figure()
plt.imshow(rgbt.astype('uint8'), origin='lower')
plt.show(block=False)
