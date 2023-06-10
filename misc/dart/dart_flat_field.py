import matplotlib.pyplot as plt
import numpy as np
# %matplotlib qt
from astro_utils import *
from astro_fill_holes import *
import os
import pandas as pd
# import cv2
# %matplotlib qt

os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/')
stuff = '/home/innereye/astro/dart/'
path = np.asarray(list_files('/home/innereye/Dropbox/Moris_20220926-20221002/', '*.fits'))
kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data
mmdd = np.asarray([x[9:13] for x in path])
mmddu = np.unique(mmdd)
hot = pd.read_csv(stuff+'hot_pixels.csv')

date = '0929'
filename = stuff + 'xy'+date+'.csv'
dfxy = pd.read_csv(filename)
empty = np.zeros((512, 512, 11), 'float')
# plt.figure()
for ii in range(11):
    hdu = fits.open('/home/innereye/Dropbox/Moris_20220926-20221002/'+dfxy['file'].loc[ii])
    d = hdu[0].data.copy().astype('float')
    d[d > 210] = np.nan
    print(np.sum(d > 210))
    empty[:,:,ii] = d

m = np.nanmean(empty, axis=2)
plt.figure()
plt.imshow(m)
    # plt.subplot(3,4,ii+1)
    # plt.imshow(d)

## get center of didymos for al images
# halfdid = 150
# data = np.zeros((halfdid*2, halfdid*2, 7))
# raw = data.copy()
noise_med = np.zeros((512, 512, 2), 'float')
noise_mean = np.zeros((512, 512, 2), 'float')
for day in range(2):
    date = mmddu[day]
    print(date)
    filt = []
    pathc = path[mmdd == date]
    filename = stuff + 'xy'+date+'.csv'
    dfxy = pd.read_csv(filename)
    # didV = np.zeros((halfdid*2, halfdid*2, 7))
    # filt = []
    pathc = path[mmdd == date]
    filename = stuff + 'xy'+date+'.csv'
    dfxy = pd.read_csv(filename)
    xy = np.zeros((len(dfxy),2), int)
    xy[:, 0] = dfxy['x'].to_numpy()
    xy[:, 1] = dfxy['y'].to_numpy()
    n = 0
    background = []
    for ii in range(len(pathc)):
        if dfxy['x'].loc[ii] > 0:
            f = pathc[ii]
            hdu = fits.open(f)
            if hdu[0].header['FILTER'][-1] == 'V':
                n += 1
                img = hdu[0].data.copy()
                img = img.astype(float)
                # for pix in range(3):
                #     x = hot['x'].loc[pix]
                #     y = hot['y'].loc[pix]
                #     img[x, y] = (img[x - 1, y] + img[x + 1, y] + img[x, y - 1] + img[x, y + 1]) / 4
                # img = img[:496, 25:]
                hd = 30
                img[xy[ii, 0] - hd: xy[ii,0] + hd, xy[ii, 1] - hd: xy[ii, 1] + hd] = np.nan
                background.append(img)
    cube = np.zeros((512, 512, n))
    for jj in range(n):
        cube[:,:,jj] = background[jj]
    med = np.nanmedian(cube, axis=2)
    noise_med[:, :, day] = med
    for jj in range(n):
        tmp = cube[:,:,jj]
        tmp[tmp > med*1.5] = np.nan
        tmp[tmp > 210] = np.nan
        tmp[tmp < 200] = 200
        cube[:,:,jj] = tmp
    mean = np.nanmean(cube, axis=2)
    noise_mean[:,:,day] = mean

plt.figure()
ii = 0
plt.subplot(2, 2, 1)
plt.imshow(noise_med[:,:,ii],origin='lower')
plt.clim(202, 203)
plt.title('day' + str(ii)+', median')
plt.subplot(2, 2, 2)
plt.imshow(noise_mean[:, :, ii], origin='lower')
plt.clim(201.7, 202)
plt.title('day' + str(ii)+', mean')
ii = 1
plt.subplot(2, 2, 3)
plt.imshow(noise_med[:,:,ii],origin='lower')
plt.clim(201, 202)
plt.title('day' + str(ii)+', median')
hotpix = 0
# plt.plot(hot['y'].loc[hotpix],hot['x'].loc[hotpix], 'or', markersize=5, markerfacecolor=None, markeredgewidth=0.1)
plt.subplot(2, 2, 4)
plt.imshow(noise_mean[:, :, ii], origin='lower')
plt.clim(201, 201.3)
plt.title('day' + str(ii)+', mean')

#
# with open(stuff+'per_day.pkl', 'wb') as f:
#     pickle.dump(data, f)
#
# with open(stuff+'per_day_raw.pkl', 'wb') as f:
#     pickle.dump(raw, f)
#
# out = cv2.VideoWriter(stuff+'per_day.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 3, (300, 300), False)
# for day in range(7):
#     out.write(ops[:,:,day])
# out.release()


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
