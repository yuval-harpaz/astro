from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os
# manifest = download_fits('ngc 628', include=['_miri_', '_nircam_', 'clear'])
dir = list_files.__code__.co_filename[:-14]
path = list_files('ngc_628', search='*nircam*.fits')
fname = [os.path.basename(ii) for ii in path]
layers = reproject(path, project_to=1)
mn = 0
mx = 5
img = layers[:, :, 1:]
img = (img-mn)/(mx-mn)*255
img[img < 0] = 0
img[img > 255] = 255
img = img.astype(np.uint8)
plt.imsave(dir+'/pics/'+'NGC_628_nircam.png', np.flipud(img))
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.subplots_adjust(right=1, left=0, top=1, bottom=0)
plt.imshow(img, origin='lower')
plt.show(block=False)
plt.axis('off')
plt.text(100, 100, 'NGC 628. NIRCam filters: R=f300m, G=f335m, B=f360m', color='w')


path = ['mastDownload/JWST/jw02107-o039_t018_miri_f770w/jw02107-o039_t018_miri_f770w_i2d.fits',
        'mastDownload/JWST/jw02107-o039_t018_miri_f1000w/jw02107-o039_t018_miri_f1000w_i2d.fits',
        'mastDownload/JWST/jw02107-o039_t018_miri_f1130w/jw02107-o039_t018_miri_f1130w_i2d.fits']
# path = list_files('ngc_628', search='*miri*.fits')
layers = reproject(path, project_to=0)
# balance the filters
mnl = np.asarray([8, 30, 42], int)
mxl = np.asarray([31, 36, 56], int)
img = layers.copy()
for ii in [0, 1, 2]:
    img[:, :, ii] = (img[:, :, ii]-mnl[ii])/(mxl[ii]-mnl[ii])*255
img[img < 0] = 0
img[img > 255] = 255
img = img.astype(np.uint8)
img = np.mean(img, axis=2)
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.subplots_adjust(right=1, left=0, top=1, bottom=0)
plt.imshow(img, origin='lower', cmap='gray')
# plt.clim(mn, 0.1)
plt.axis('off')
plt.text(500, 50, 'NGC 628. miri filters: f770w, f1000w, f1130w', color='w',weight='bold')
plt.show(block=False)
plt.imsave(dir+'/pics/'+'NGC_628_miri.png', np.flipud(img))

