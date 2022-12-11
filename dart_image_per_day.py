import matplotlib.pyplot as plt
import numpy as np
# %matplotlib qt
from astro_utils import *
from astro_fill_holes import *
import os
import pandas as pd
import cv2
# %matplotlib qt

os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/')
stuff = '/home/innereye/astro/dart/'
path = np.asarray(list_files('/home/innereye/Dropbox/Moris_20220926-20221002/', '*.fits'))
kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data
mmdd = np.asarray([x[9:13] for x in path])
mmddu = np.unique(mmdd)
## get center of didymos for al images
halfdid = 150
data = np.zeros((halfdid*2, halfdid*2, 7))
raw = data.copy()
ops = np.zeros((halfdid*2, halfdid*2, 7), 'uint8')
for day in range(7):
    date = mmddu[day]
    print(date)
    filt = []
    pathc = path[mmdd == date]
    filename = stuff + 'xy'+date+'.csv'
    dfxy = pd.read_csv(filename)
    hot = pd.read_csv(stuff+'hot_pixels.csv')

    didV = np.zeros((halfdid*2, halfdid*2, 7))
    filt = []
    pathc = path[mmdd == date]
    filename = stuff + 'xy'+date+'.csv'
    dfxy = pd.read_csv(filename)
    xy = np.zeros((len(dfxy),2), int)
    xy[:, 0] = dfxy['x'].to_numpy()
    xy[:, 1] = dfxy['y'].to_numpy()
    n = 0
    did_clean = []
    for ii in range(len(pathc)):
        if dfxy['x'].loc[ii] > 0:
            f = pathc[ii]
            hdu = fits.open(f)
            if hdu[0].header['FILTER'][-1] == 'V':
                n += 1
                img = hdu[0].data.copy()
                img = img.astype(float)
                for pix in range(3):
                    x = hot['x'].loc[pix]
                    y = hot['y'].loc[pix]
                    img[x, y] = (img[x - 1, y] + img[x + 1, y] + img[x, y - 1] + img[x, y + 1]) / 4
                # img = img[:496, 25:]
                did_clean.append(img[xy[ii, 0] - halfdid:xy[ii, 0] + halfdid,
                                      xy[ii, 1] - halfdid:xy[ii, 1] + halfdid])
    cube = np.zeros((halfdid*2, halfdid*2, n))
    for jj in range(n):
        if did_clean[jj].shape == (300,300):
            cube[:,:,jj] = did_clean[jj]
        else:
            cube[:,:,jj] = np.nan
    med = np.nanmedian(cube, axis=2)
    for jj in range(n):
        tmp = cube[:,:,jj]
        tmp[tmp > med*1.5] = np.nan
        cube[:,:,jj] = tmp
    # ignore smeared asteroid
    noise = cube.copy()
    noise[120:180,120:180,:] = np.nan
    ns = np.zeros(cube.shape[2])
    for ii in range(noise.shape[2]):
        ns[ii] = np.sum(noise[:,:,ii] > 210)
        if ns[ii] > 15:
            cube[:,:,ii] = np.nan
# noise = np.nansum(np.nansum(noise, axis=0), axis=0)

    d = np.nanmean(cube, axis=2)
    raw[:, :, day] = d.copy()
    d = convolve(d, kernel=kernel.array)
    data[:, :, day] = d
    if day == 4:
        d = d*1.5
    op = (d - np.mean(d[6:-7, 30])) / 2 * 255
    op[op < 0] = 0
    op[op > 255] = 255
    op = op.astype('uint8')
    ops[:, :, day] = op

with open(stuff+'per_day.pkl', 'wb') as f:
    pickle.dump(data, f)

with open(stuff+'per_day_raw.pkl', 'wb') as f:
    pickle.dump(raw, f)

out = cv2.VideoWriter(stuff+'per_day.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 3, (300, 300), False)
for day in range(7):
    out.write(ops[:,:,day])
out.release()


# didV[:,:, dd] = np.nanmean(cube, axis=2)
# frames = int(cube.shape[2]/10)-1
# knl = Gaussian2DKernel(x_stddev=3)
# out = cv2.VideoWriter(stuff+'per_day.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (300, 300), False)
# for ii in range(cube.shape[2]):
#     d = cube[:,:,ii].copy()
#     d[np.isnan(d)] = 0
#     # d = convolve(d, kernel=knl.array)
#     op = (d - np.mean(d[6:-7, 30])) / 2 * 255
#     if ii == 4:
#         op = op * 1.5
#     op[op < 0] = 0
#     op[op > 255] = 255
#     op = op.astype('uint8')
#     if np.sum(op) > 0:
#         out.write(op)
#     else:
#         print('frame'+str(ii))
# out.release()
#
# # plt.figure()
# # for ii in range(7):
# #     plt.subplot(2, 4, ii+1)
# #     plt.imshow(didV[:,:,ii] + 202-np.median(didV[:,30,ii]), cmap='gray', origin='lower')
# #     plt.clim(200, 205)
# #     plt.axis('off')
# #
# # knl = Gaussian2DKernel(x_stddev=1)
# # plt.figure()
# # for ii in range(7):
# #     frame = didV[:, :, ii]
# #     sm = convolve(frame, kernel=knl.array)
# #     op = (sm - np.mean(sm[6:-7, 30])) / 2 * 255
# #     if ii == 4:
# #         op = op*1.5
# #     op[op < 0] = 0
# #     op[op > 255] = 255
# #     op = op.astype('uint8')
# #     plt.subplot(2, 4, ii + 1)
# #     plt.imshow(op, cmap='gray', origin='lower')
# #     plt.axis('off')
# #     plt.imsave(stuff + '00' + str(ii) + '_Vnorm.png', op, cmap='gray', origin='lower')
# #
# #
# #
#
#
