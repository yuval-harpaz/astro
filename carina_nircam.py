from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os
os.chdir('/home/innereye/JWST/Carina/')
path = ['jw02731-o001_t017_nircam_clear-f335m_i2d.fits',
        'jw02731-o001_t017_nircam_clear-f444w_i2d.fits',
        'jw02731-o001_t017_nircam_f444w-f470n_i2d.fits']
# path = list_files('ngc_628', search='*miri*.fits')
layers = reproject(path, project_to=0)

mnl = np.asarray([7.9, 19, 32.5], int)
# mxl = np.asarray([45, 100, 275, 38],int)
mxl = np.asarray([11, 20, 33], int)
# mxl = mxl-0.5*(mxl-mnl)
mnl = [0,3.5,0]
mxl = [17,10,20]
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
img1 = np.mean(img, axis=2)

fig, ax = plt.subplots(nrows=1, ncols=1)
plt.subplots_adjust(right=1, left=0, top=1, bottom=0)
plt.imshow(img, origin='lower')
# plt.clim(mn, 0.1)
plt.axis('off')
# plt.text(500, 100, 'NGC 628. miri filters: R=f770w, G=f1000w, B=f1130w', color='w',weight='bold')
plt.clim(10, 255)
plt.show(block=False)
zeros = np.min(layers, axis=2) == 0
img_fix = img.copy()
for ll in range(3):
    layer = img[:, :, ll]
    for ii, row in enumerate(layer):
        # loc = find_peaks(255-row, prominence=120, width=[1,20])[0]
        start = np.where((row[:-1] > 0) & (row[1:] == 0))[0]+1
        end = np.where((row[1:] > 0) & (row[:-1] == 0))[0]
        if len(start) > 0 and len(end) > 0:
            if end[-1] >= start[0]:
                end = end[end >= start[0]]
                start = start[start <= end[-1]]
                if len(start) == len(end):
                    for zz in range(len(start)):
                        if (0 < (end[zz] - start[zz]) < 30) and img[ii, start[zz]-2, ll] > 240 and img[ii, end[zz]+2, ll] > 240:
                            img_fix[ii, start[zz]:end[zz], ll] = 255
                            img_fix[ii, start[zz]:end[zz]+1, ll] = np.linspace(img[ii, start[zz]-1, ll], img[ii, end[zz]+2, ll], end[zz]-start[zz]+1)
                        # else:
                        #     print('?')
                else:
                    raise ValueError('start and end do not match')


    # if len(loc) > 0:
    #     for jj in loc:
    #         neib = np.where(row > 240)[0]
    #         if len(neib) > 0:
    #             if (np.min(neib) < jj < np.max(neib)) and jj > 40 and jj < img1.shape[1]-20:
    #                 neib0 = neib[neib < jj][-1]
    #                 neib1 = neib[neib > jj][0]
    #                 if (jj - neib0 < 50) and (neib1 - jj < 50):
    #                     holes[ii, neib0:neib1] = True
    #                 # col = img1[:, jj]
    #                 # neib = np.where(col > 200)[0]
    #                 # if (np.min(neib) < ii < np.max(neib)) and ii > 50 and ii < img1.shape[1]-50:
    #                 #     neib0 = neib[neib < ii][-1]
    #                 #     neib1 = neib[neib > ii][0]
    #                 #     if (ii - neib0 < 50) and (neib1 - ii < 50):
    #                 #         holes[neib0:neib1, jj] = True

# img1[holes] = 255
plt.figure()
plt.imshow(img_fix, origin='lower')
plt.axis('off')
plt.show(block=False)

plt.imsave('carina_nircam1.png', np.flipud(img_fix))

#
# plt.plot(img1[314,:])
# plt.show(block=False)
# mnl = []
# mxl = []
# for ii in range(layers.shape[2]):
#     # mnl.append(np.nanpercentile(layers[:, :, ii], 0.1))
#     # mxl.append(np.nanpercentile(layers[:, :, ii], 99.99))
#     mnl.append(np.median(layers[:, :, ii]/2))
#     mxl.append(np.median(layers[:, :, ii]*5))
# mnl = 2*np.asarray(mnl)
# mxl = 3*np.asarray(mnl)
#
#
#
# plt.figure()
# for ii in range(layers.shape[2]):
#     plt.subplot(2, 2, ii+1)
#     plt.imshow(layers[:, :, ii], cmap='hot')
#     plt.clim(mnl[ii], mxl[ii])
#     plt.show(block=False)
#     plt.axis('off')
#     plt.title(fname[ii])

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
