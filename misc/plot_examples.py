from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os
from astro_utils import *
dir = list_files.__code__.co_filename[:-14]
path = list_files('ngc_628', search='*nircam*.fits')
# get filename from full path
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
# plt.clim(mn, 0.1)
plt.show(block=False)
plt.axis('off')
plt.text(100, 100, 'NGC 628. NIRCam filters: R=f300m, G=f335m, B=f360m', color='w')


path = ['mastDownload/JWST/jw02107-o039_t018_miri_f770w/jw02107-o039_t018_miri_f770w_i2d.fits',
        'mastDownload/JWST/jw02107-o039_t018_miri_f1000w/jw02107-o039_t018_miri_f1000w_i2d.fits',
        'mastDownload/JWST/jw02107-o039_t018_miri_f1130w/jw02107-o039_t018_miri_f1130w_i2d.fits']
# path = list_files('ngc_628', search='*miri*.fits')
layers = reproject(path, project_to=0)

mnl = np.asarray([8, 30, 42], int)
# mxl = np.asarray([45, 100, 275, 38],int)
mxl = np.asarray([31, 36, 56], int)
# mxl = mxl-0.5*(mxl-mnl)

img = layers.copy()
for ii in [0, 1, 2]:
    img[:, :, ii] = (img[:, :, ii]-mnl[ii])/(mxl[ii]-mnl[ii])*255
    # img[img < 0] = 0
    # img[img > 255] = 255
    # plt.figure()
    # plt.imshow(img[:, :, ii])
    # plt.axis('off')
    # plt.show(block=False)
    # img = (img-mn)/(mx-mn)*255
img[img < 0] = 0
img[img > 255] = 255
img = img.astype(np.uint8)
img = np.mean(img, axis=2)
fig, ax = plt.subplots(nrows=1, ncols=1)
plt.subplots_adjust(right=1, left=0, top=1, bottom=0)
plt.imshow(img, origin='lower', cmap='gray')
# plt.clim(mn, 0.1)
plt.axis('off')
plt.text(500, 100, 'NGC 628. miri filters: R=f770w, G=f1000w, B=f1130w', color='w',weight='bold')
plt.show(block=False)
plt.imsave(dir+'/pics/'+'NGC_628_miri.png', np.flipud(img))



mnl = []
mxl = []
for ii in range(layers.shape[2]):
    # mnl.append(np.nanpercentile(layers[:, :, ii], 0.1))
    # mxl.append(np.nanpercentile(layers[:, :, ii], 99.99))
    mnl.append(np.median(layers[:, :, ii]/2))
    mxl.append(np.median(layers[:, :, ii]*5))
mnl = 2*np.asarray(mnl)
mxl = 3*np.asarray(mnl)



plt.figure()
for ii in range(layers.shape[2]):
    plt.subplot(2, 2, ii+1)
    plt.imshow(layers[:, :, ii], cmap='hot')
    plt.clim(mnl[ii], mxl[ii])
    plt.show(block=False)
    plt.axis('off')
    plt.title(fname[ii])

# path = list_files('cartwheel', search='*.fits', exclude='clear')
# median = mosaic(path, plot=True, method='median')
# mn = 0.11
# mx = 1.7
# plt.clim(mn, mx)
# plt.show(block=False)
# img = (median-mn)/(mx-mn)*255
# img[img < 0] = 0
# img[img > 255] = 255
# img = img.astype(np.uint8)
# img[1947:1952, 1019:1024] = 255
# img[775:777, 2042:2045] = 255
# plt.imsave('median.png', img.T, cmap='gray')
# plt.imsave('median_hot.png', img.T, cmap='hot')
# layers = mosaic(path, plot=False, method='layers')
# layers[layers == 0] = np.nan
# rgb = np.zeros((median.shape[0], median.shape[1], 3))
# for ii in range(3):
#     layer1 = np.nanmedian(layers[:, :, ii*8:ii*8+8], axis=2)
#     layer1[np.isnan(layer1)] = np.nanmedian(layer1)
#     rgb[:, :, ii] = layer1
# img = (rgb - mn) / (mx - mn) * 255
# img[img < 0] = 0
# img[img > 255] = 255
# img = img.astype(np.uint8)
# img[1945:1952, 1018:1025] = 255
# img[773:777, 2042:2046] = 255
# plt.imsave('median_rgb.png', img.swapaxes(0, 1))
# print('images saved here: '+os.getcwd())
# print('Buttonpress the figure to kill program')
# plt.waitforbuttonpress()
#
