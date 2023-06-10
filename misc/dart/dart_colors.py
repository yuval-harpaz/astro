import matplotlib.pyplot as plt
import numpy as np
# %matplotlib qt
from astro_utils import *
# from astro_fill_holes import *
import os
import pandas as pd
from scipy.ndimage import median_filter
# import cv2
# %matplotlib qt

os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/')
stuff = '/home/innereye/astro/dart/'
df = pd.read_csv(stuff+'meta.csv')
mmdd = np.asarray([x[9:13] for x in df['file']])
filt = np.asarray([x[-1].lower() for x in df['filter']])

# path = np.asarray(list_files('/home/innereye/Dropbox/Moris_20220926-20221002/', '*.fits'))
kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data
# mmdd = np.asarray([x[9:13] for x in path])
# mmddu = np.unique(mmdd)
## get center of didymos for al images
halfdid = 150
data = {'0928': {}, '0929': {}, '1002': {}}
for dd, date in enumerate(['0928', '0929', '1002']):
    for ff in ['r','g','v']:
        idx = np.where((filt == ff) & (mmdd == date) & (df['x'] > 0) & (df['y'] < 370))[0]
        dat = np.zeros((300,300, len(idx)))
        for c, row in enumerate(idx):
            f = '/home/innereye/Dropbox/Moris_20220926-20221002/' + df['file'][row]
            hdu = fits.open(f)
            img = hdu[0].data.copy()
            x = df['x'][row]
            y = df['y'][row]
            dat[:, :, c] = img[x-halfdid:x+halfdid, y-halfdid:y+halfdid]
        data[date][ff] = dat
##
c = 0
plt.figure()
for date in ['0928','0929','1002']:
    for ff in ['r','g','v']:
        c += 1
        dat = data[date][ff].copy()
        med = np.median(dat, axis=2)
        for ii in range(dat.shape[2]):
            layer = dat[:,:,ii]
            noise = layer.copy()
            noise[ 110:190, 110:210] = 0
            if np.sum(noise > 210) > 25:
                layer[:, :] = np.nan
            else:
                layer[layer > med*1.5] = np.nan
                # layer[layer > 210] = np.nan
        clean = np.nanmean(dat, axis=2)
        plt.subplot(3, 3, c)
        plt.imshow(clean, origin='lower')
        plt.clim(200, 210)
        plt.axis('off')
        good = int(np.median(np.nansum(~np.isnan(dat), axis=2)))
        plt.title(date+' '+str(good) + '/' + str(dat.shape[2]) + ' ' + ff)
##
means = data.copy()
for date in ['0928','0929','1002']:
    for ff in ['r','g','v']:
        c += 1
        dat = data[date][ff].copy()
        med = np.median(dat, axis=2)
        for ii in range(dat.shape[2]):
            layer = dat[:,:,ii]
            noise = layer.copy()
            noise[ 110:190, 110:210] = 0
            if np.sum(noise > 210) > 25:
                layer[:, :] = np.nan
            else:
                layer[layer > med*1.5] = np.nan
                # layer[layer > 210] = np.nan
        clean = np.nanmean(dat, axis=2)
        means[date][ff] = clean
##
c = 0
plt.figure()
for date in ['0928','0929','1002']:
    for ff in ['r','g']:
        c += 1
        plt.subplot(3, 2, c)
        plt.imshow(means[date][ff] - means[date]['v'], origin='lower')
        plt.clim(-10, 10)
        plt.axis('off')
        good = int(np.median(np.nansum(~np.isnan(dat), axis=2)))
        plt.title(date + ' ' + ff+'-v')


##
log = False
c = 0
plt.figure()
for date in ['0928','0929','1002']:
    c += 1
    plt.subplot(2, 2, c)
    r = means[date]['r'].copy()
    mask = r < np.median(r)+1.5
    r = r - np.mean(r[:50,:50])
    # r = convolve(r, kernel=kernel)
    g = means[date]['g'].copy()
    g = g - np.mean(g[:50, :50])
    # g = convolve(g, kernel=kernel)
    # ratio = convolve(r/g, kernel=kernel)
    ratio = median_filter(r/g, footprint=kernel.array)
    # ratio = r/g
    ratio[mask] = 1
    if log:
        ratio[ratio < 1] = 1
        plt.imshow(np.log(ratio), origin='lower')
        plt.clim(0.2, 0.75)
    else:
        plt.imshow(ratio, origin='lower')
        plt.clim(1.5, 2.1)
    plt.axis('off')
    plt.title(date)
plt.subplot(2,2,4)
# plt.imshow(ratio, origin='lower')
# plt.clim(1, 2)
plt.axis('off')
plt.colorbar()