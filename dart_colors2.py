
# %matplotlib qt
from astro_utils import *
import os
import pandas as pd
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt
import numpy as np

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
means = data.copy()
for date in ['0928','0929','1002']:
    print(date)
    for ff in ['r','g','v']:
        print(ff)
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
        print(dat.shape[2])
# ##
# c = 0
# plt.figure()
# for date in ['0928','0929','1002']:
#     for ff in ['r','g']:
#         c += 1
#         plt.subplot(3, 2, c)
#         plt.imshow(means[date][ff] - means[date]['v'], origin='lower')
#         plt.clim(-10, 10)
#         plt.axis('off')
#         good = int(np.median(np.nansum(~np.isnan(dat), axis=2)))
#         plt.title(date + ' ' + ff+'-v')


##
log = False
c = 0
plt.figure()
for date in ['0928', '0929', '1002']:
    c += 1
    plt.subplot(2, 2, c)
    r = means[date]['r'].copy()
    mask = r < np.median(r)+1.5
    # r = r - np.mean(r[:50,:50])
    g = means[date]['g'].copy()
    # g = g - np.mean(g[:50, :50])
    # g = convolve(g, kernel=kernel)
    # ratio = convolve(r/g, kernel=kernel)
    ratio = median_filter(r/g, footprint=kernel.array)
    # ratio = r/g
    # ratio[mask] = 1
    if log:
        ratio[ratio < 1] = 1
        plt.imshow(np.log(ratio), origin='lower')
        plt.clim(0.2, 0.75)
    else:
        plt.imshow(ratio)
        plt.clim(1, 1.02)
    plt.axis('off')
    plt.title(date)
plt.subplot(2, 2, 4)
plt.axis('off')
plt.colorbar()

##
date_str = ['Sep 28','Sep 29','Oct 02']
levs = 2.0**np.arange(0,7,0.5)
levs = levs[:-1]
cmap = 'jet'
c = 0
plt.figure()
for date in ['0928', '0929', '1002']:
    c += 1
    plt.subplot(2, 2, c)
    r = means[date]['r'].copy()
    # r = r - np.mean(r[:50,:50])
    r = convolve(r, kernel=kernel)
    mask = r < 0.5  # np.median(r) + 1.5
    g = means[date]['g'].copy()
    # g = g - np.mean(g[:50, :50])
    g = convolve(g, kernel=kernel)
    per_nm = (r / g) ** (1/(616.6 - 468.6))
    # per_nm[np.isnan(per_nm)] = 1
    # slope = (r/g)/(616.6-468.6)*100*100
    slope = per_nm ** 100 * 100 - 100
    # slope = median_filter(slope, footprint=kernel.array)
    # ratio = r/g
    # slope[mask] = 1
    slope[slope < 1] = 1
    cs = plt.contourf(np.flipud(slope), levs, norm=matplotlib.colors.LogNorm(), cmap=cmap)
    # plt.imshow(slope) #  norm=matplotlib.colors.LogNorm(vmin=10, vmax=100)
    # plt.clim(1, 12)
    plt.axis('off')
    plt.axis('square')
    plt.xlim(100, 200)
    plt.ylim(100, 200)
    plt.title(date_str[c-1])
    print(slope.max())
plt.subplot(2, 2, 4)
cs = plt.contourf(slope, levs, norm=matplotlib.colors.LogNorm(), cmap=cmap)
plt.axis('off')
plt.axis('square')
plt.xlim(100, 200)
plt.ylim(100, 200)
plt.colorbar()

##
c = 0
plt.figure()
for date in ['0928', '0929', '1002']:
    c += 1
    plt.subplot(2, 2, c)
    r = means[date]['r'].copy()
    r = convolve(r, kernel=kernel)
    g = means[date]['g'].copy()
    g = convolve(g, kernel=kernel)
    per_nm = (r / g) ** (1/(616.6 - 468.6))
    slope = per_nm ** 100 * 100 - 100
    print(slope.max())
    slope[slope < 1] = 1
    plt.imshow(slope)
    # plt.clim(1, 12)
    plt.axis('off')
    plt.axis('square')
    plt.xlim(100, 200)
    plt.ylim(100, 200)
    plt.title(date)
# plt.subplot(2, 2, 4)
# cs = plt.contourf(slope, levs, norm=matplotlib.colors.LogNorm(), cmap=cmap)
# plt.axis('off')
# plt.axis('square')
# plt.xlim(100, 200)
# plt.ylim(100, 200)
# plt.colorbar()

ps = [51.14, 39.88, 23.07]
plt.figure()
plt.bar([0, 1, 4], ps)
plt.title('spectral slope')
plt.ylabel('slope (%)')
plt.xticks([0,1,4], date_str)