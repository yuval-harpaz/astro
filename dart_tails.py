%matplotlib qt

# import matplotlib.pyplot as plt
# import numpy as np
from astro_utils import *
from astro_fill_holes import *
import os
from cv2 import circle
# import pandas as pd
# import pickle
## Take the first day data (before impact) to compute flat field
os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/')
stuff = '/home/innereye/astro/dart/'
path = list_files('/home/innereye/Dropbox/Moris_20220926-20221002/', '*.fits')
# kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data
mmdd = np.asarray([x[9:13] for x in path])
mmddu = np.unique(mmdd)

## parameters and baseline correction
gray = False
save = False
smooth = False
zoom = False
clim1 = 1
height = 0.07
tang_smooth = 10
if smooth:
    data = np.load(stuff + 'per_day_smooth.pkl', allow_pickle=True)
    max_rad = 150  # 145
    prom = 0.005

else:
    data = np.load(stuff + 'per_day_raw.pkl', allow_pickle=True)
    max_rad = 212
    prom = 0.1
    height = 0.1
for day in range(7):
    layer = data[:, :, day]
    bl = np.percentile(layer, 25)
    layer = layer - bl
    if day == 4:
        layer = layer * 1.5
    data[:, :, day] = layer
## loop
plt.figure()
for day in range(7):
    peaks = []
    for rad in np.arange(2, max_rad, 1):
        img1 = circle(np.zeros((300,300,3)), (150,150), rad, (1,0,0), 1)[:,:,0]
        x, y = np.where(img1[:,:151])
        idx = np.arange(len(x))
        xu = np.unique(x)
        for ii in range(int(len(xu)/2)):
            xx = xu[ii]
            rows = np.where(x == xx)[0]
            if len(rows) > 1:
                yy = y[rows]
                order = np.argsort(-yy)
                idx[rows] = idx[rows[order]]
        y = y[idx]
        x1 = np.flipud(x)
        xc = np.asarray(list(x)+list(x1[1:-1]))
        yc = np.asarray(list(y)+list(300-y[1:-1]-1))
        lc = data[xc,yc,day]
        lc = np.asarray(list(lc) + list(lc[:10]))
        if smooth:  # already smoothed data
            dist = 10
            lcs = np.squeeze(movmean(lc, tang_smooth))
            # lcs = lc
        else:  # tangential smoothing
            dist = int(len(xc) / 8)
            lcs = np.squeeze(movmean(lc,tang_smooth))
        pks = find_peaks(lcs, distance=dist, prominence=prom, height=height)[0]  # , width=4
        pks = pks[pks > 5]
        pks[pks > len(xc)-1] = pks[pks > len(xc)-1] - len(xc)
        pks = np.unique(pks)
        for ipk in range(len(pks)):
            peaks.append([xc[pks[ipk]],yc[pks[ipk]]])
    peakmap = data[:,:,day].copy()
    mx = peakmap.max()
    for peak in peaks:
        if peakmap[peak[0],peak[1]] < clim1/2:
            peakmap[peak[0],peak[1]] = 1000
        else:
            peakmap[peak[0], peak[1]] = -1000
    plt.subplot(2,4, day+1)
    if zoom:
        img2 = peakmap[75:225, 75:225]
        # orig = data[75:225, 75:225, day]
    else:
        img2 = peakmap
    img = np.zeros((img2.shape[0],img2.shape[1], 3))
    img[:, :, 0] = img2
    for rgb in range(3):
        img[:, :, rgb] = img2
    if not gray:
        im = img[:, :, 0].copy()
        im[np.abs(im) == 1000] = 1000
        img[:, :, 0] = im
        im = img[:, :, 1].copy()
        im[np.abs(im) == 1000] = -1000
        img[:, :, 1] = im
        im = img[:, :, 2].copy()
        im[np.abs(im) == 1000] = -1000
        img[:, :, 2] = im
    img[img > clim1] = clim1
    img[img < 0] = 0
    plt.imshow(img, origin='lower')
    plt.clim(0, clim1)
    plt.axis('off')
    plt.title(mmddu[day].replace('10','Sep ').replace('10','Oct '))
    if save:
        op = peakmap.copy()
        op = op/clim1*255
        op[op > 255] = 255
        op[op < 0] = 0
        op = op.astype('uint8')
        plt.imsave(stuff+'tails/peaks0'+str(day)+'.jpg',op, origin='lower', cmap='gray')


##
# img1[150 - rad, 150] = 0
# img1[150 + rad, 150] = 0
# img1[150, 150 - rad] = 0
# img1[150, 150 + rad] = 0
# img1[150 - rad + 1, 150] = 1
# img1[150 + rad - 1, 150] = 1
# img1[150, 150 - rad + 1] = 1
# img1[150, 150 + rad - 1] = 1

# plt.plot(xc, yc)
# plt.figure()
# plt.plot(lc, label='raw')
# plt.plot(lcs, label='smooth')
# plt.plot(pks, lcs[pks],'.r', label='peaks')




# order = np.argsort(x)
# x = x[order]
# y = y[order]
#
#
#
# jet = matplotlib.cm.get_cmap('jet', 9)
# jet = jet(np.linspace(0, 1, 9))
#
# plt.plot(data[150, :,:]);plt.legend(mmddu)
#
# plt.figure()
# for ii in range(7):
#     plt.plot(data[150,:,ii]-np.mean(data[150,22:30,ii]), label=mmddu[ii], color=jet[ii+1,:3])
# plt.xlim(120, 180)
# plt.ylim(0, 300)
# plt.grid()
# plt.legend()
#
# plt.figure()
# for ii in range(7):
#     d = data[:,:,ii]
#     d = d - np.mean(d[6:-7, 30])
#     if ii == 4:
#         d = d*1.5
#     op = d / 2 * 255
#     op[op < 0] = 0
#     op[op > 255] = 255
#     op = op.astype('uint8')
#     plt.subplot(2,4,ii+1)
#     plt.imshow(op, origin='lower', cmap='gray')
#
# levs = 10.0**np.arange(-2,2.6,0.5)
# plt.figure()
# for ii in range(7):
#     d = data[:,:,ii].copy()
#     bl = (np.median(d[15:50,15:50])+np.median(d[-50:-15,-50:-15]))/2
#     d = d - bl
#     if ii == 4:
#         d = d*1.5
#     d[d < 0.01] = 0.01
#     # op = d / 2 * 255
#     # op[op < 0] = 0
#     # op[op > 255] = 255
#     # op = op.astype('uint8')
#     plt.subplot(2,4,ii+1)
#     # fig, ax = plt.subplots()
#     # lev_exp = np.arange(-10, 100)
#     #                     levs = np.power(10, lev_exp)
#     #                     cs = ax.contourf(X, Y, z, levs, norm=colors.LogNorm())
#     # cs = plt.contourf(d, locator=matplotlib.ticker.LogLocator(base=5, numdecs=5))
#     cs = plt.contourf(d, levs, norm = matplotlib.colors.LogNorm())
#     plt.axis('off')
#     plt.axis('square')
# plt.subplot(2,4,8)
# plt.axis('off')
# plt.colorbar(format='%.2f')
#     # cs = plt.contourf(d, levels=[0, 10, 20, 30, 40])
