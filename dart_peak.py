import matplotlib.pyplot as plt
import numpy as np
%matplotlib qt
from astro_utils import *
from astro_fill_holes import *
import os
import pandas as pd

os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/')
stuff = '/home/innereye/astro/dart/'
path = np.asarray(list_files('/home/innereye/Dropbox/Moris_20220926-20221002/', '*.fits'))
kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data
mmdd = np.asarray([x[9:13] for x in path])
mmddu = np.unique(mmdd)
## get center of didymos for al images
plt.figure()
dd = 0
for date in mmddu:
    pathc = path[mmdd == date]
    filename = stuff + 'xy'+date+'.csv'
    if os.path.isfile(filename):
        tmp = pd.read_csv(filename, header=None)
        if len(tmp.columns) == 2:
            dfxy = pd.DataFrame(list(pathc), columns=['file'])
            dfxy['x'] = tmp[0]
            dfxy['y'] = tmp[1]
            dfxy.to_csv(filename)
        else:
            dfxy = pd.read_csv(filename)
    else:
        print('XXXXXXXXXXXX    '+date+'    XXXXXXXXXXXXXX')
        dfxy = pd.DataFrame(list(pathc), columns=['file'])
        xy = np.zeros((len(pathc), 2), int)
        for ii in range(len(pathc)):
            f = pathc[ii]
            hdu = fits.open(f)
            img = hdu[0].data.copy()
            img = img.astype(float)
            img[:120, :] = 200
            img[360:, :] = 200
            img[:, :120] = 200
            img[:, 360:] = 200
            # img = img[:496, 25:]
            # smooth data before looking for max to avoid stars / noise
            smoothed = convolve(img, kernel=kernel.array)
            xyc = np.unravel_index(smoothed.argmax(), smoothed.shape)  # current x y coordinates of asteroid
            xy[ii, :] = [xyc[0], xyc[1]]
            print(ii)
        dfxy['x'] = xy[:,0]
        dfxy['y'] = xy[:,1]
        dfxy.to_csv(filename)
    plt.subplot(3,3,dd+1)
    plt.plot(dfxy['x'], dfxy['y'],'.')
    plt.title(date)
    plt.grid()
    plt.axis('square')
    plt.xlim(0, 512)
    plt.ylim(0, 512)
    plt.xticks([0, 128,256,384, 512])
    plt.yticks([0, 128, 256, 384, 512])
    dd += 1


## issue with 0929
date = '0929'
pathc = path[mmdd == date]
filename = stuff + 'xy' + date + '.csv'
print('XXXXXXXXXXXX    ' + date + '    XXXXXXXXXXXXXX')
dfxy = pd.DataFrame(list(pathc), columns=['file'])
xy = np.zeros((len(pathc), 2), int)
v = []
filt = []
for ii in range(len(pathc)):
    f = pathc[ii]
    hdu = fits.open(f)
    filt.append(hdu[0].header['FILTER'])
    img = hdu[0].data.copy()
    if img.max() > 210:
        img = img.astype(float)
        img[:120, :] = 200
        img[460:, :] = 200
        img[:, :120] = 200
        img[:, 460:] = 200
        # img = img[:496, 25:]
        # smooth data before looking for max to avoid stars / noise
        smoothed = convolve(img, kernel=kernel.array)
        xyc = np.unravel_index(smoothed.argmax(), smoothed.shape)  # current x y coordinates of asteroid
        v.append(hdu[0].data[xyc[0], xyc[1]])
        xy[ii, :] = [xyc[0], xyc[1]]
    else:
        xy[ii, :] = [0, 0]
        v.append(img.max())
    print(ii)
xy[9,:] = 0


plt.figure()
for jj in range(len(filt)):
    if filt[jj][-1] == 'V':
        co = 'gv'
    elif filt[jj][-1] == 'i':
        co = 'k.'
    elif filt[jj][-1] == 'z':
        co = 'c.'
    elif filt[jj][-1] == 'r':
        co = '.r'
    elif filt[jj][-1] == 'g':
        co = '.g'
    plt.plot(jj, v[jj],co)
plt.ylabel('maximum value')
plt.ylabel('image number')

dfxy['x'] = xy[:, 0]
dfxy['y'] = xy[:, 1]
dfxy.to_csv(filename)
plt.subplot(3, 3, dd + 1)
plt.plot(dfxy['x'], dfxy['y'], '.')
plt.title(date)
plt.grid()
plt.axis('square')
plt.xlim(0, 512)
plt.ylim(0, 512)
plt.xticks([0, 128, 256, 384, 512])
plt.yticks([0, 128, 256, 384, 512])
dd += 1


# np.savetxt(stuff + 'xy1002.csv', xy, delimiter=',', fmt='%d')
##  Get indices of hot pixels
# xy = np.genfromtxt(stuff+'xy1002.csv', delimiter=',').astype(int)  # created with dart1
# df = pd.read_csv(stuff+'hot_pixels.csv')
# didbl = np.genfromtxt(stuff+'Didymos0926.csv', delimiter=',')
#
# did_clean = np.zeros((300, 300, len(path)))
# # fix hot pixels
# for ii in range(len(path)):
#     f = path[ii]
#     hdu = fits.open(f)
#     img = hdu[0].data.copy()
#     img = img.astype(float)
#     for pix in range(3):
#         x = df['x'].loc[pix]
#         y = df['y'].loc[pix]
#         img[x,y] = (img[x-1,y] + img[x+1,y] + img[x, y-1] + img[x, y+1])/4
#     # img = img[:496, 25:]
#     did_clean[:, :, ii] = img[xy[ii,0]-halfdid:xy[ii,0]+halfdid,xy[ii,1]-halfdid:xy[ii,1]+halfdid]
#
# did_clean_med = np.median(did_clean, axis=2)
# for ii in range(did_clean.shape[-1]):
#     tmp = did_clean[:,:,ii]
#     tmp[tmp > did_clean_med * 2] = np.nan
#
#
# m = np.nanmean(did_clean, axis=2)
# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(didbl, origin='lower', cmap='gray')
# plt.clim(201, 203)
# plt.axis('off')
# plt.title('Sep 26')
# plt.subplot(1,3,2)
# plt.imshow(m, origin='lower', cmap='gray')
# plt.clim(201, 203)
# plt.axis('off')
# plt.title('Oct 2')
# plt.subplot(1,3,3)
# plt.imshow(m - didbl, origin='lower', cmap='jet')
# plt.clim(-1, 3)
# plt.axis('off')
# plt.title('Oct 2 - Sep 26')
#
# plt.imshow(m-didbl)
# plt.clim(-2, 2)
#
# for ii in range(did.shape[-1]):
#     tmp = did_clean[:,:,ii]
#     tmp[tmp > did_clean_med * 1.5] = np.nan
# did_mean = np.nanmean(did_clean, axis=2)
# plt.figure()
# plt.imshow(did_mean, origin='lower')
# plt.clim(200, 205)
# plt.show()
# # np.savetxt(stuff + 'Didymos0926.csv', did_mean, delimiter=',', fmt='%1.6f')
#
# bg_clean_med = np.nanmedian(bg_clean, axis=2)
# bg_cleanless = bg_clean.copy()
# for ii in range(did.shape[-1]):
#     tmp = bg_cleanless[:,:,ii]
#     tmp[tmp > bg_clean_med * 1.5] = np.nan
#
# bg_mean = np.nanmean(bg_cleanless, axis=2)
# # np.savetxt(stuff + 'flat_field.csv', bg_mean, delimiter=',', fmt='%1.6f')
#
# #
# # plt.figure()
# # for ii in range(len(hotx)):
# #     plt.subplot(4, 4, ii+1)
# #     plt.plot(bg[hotx[ii]+1, hoty[ii], :])
# #     plt.plot(bg[hotx[ii], hoty[ii], :])
# #     plt.ylim(195, 210)
# #     plt.plot([0,467],[202,202],'k')
# #     plt.title(str(hotx[ii])+','+str(hoty[ii]))
# #     plt.xticks([])
# #     if ii == 0:
# #         plt.legend(['control pixel','hot pixel','median'])
# # plt.suptitle('hot pixels behavior for 26 Nov data')
# # plt.show()
# #
# # plt.figure()
# # plt.imshow(np.mean(did, axis=2), origin='lower')
# # plt.clim(200, 205)
# # plt.show()
# #
# #
# # did_clean = did.copy()
# # for ii in range(3):
# #     did_clean[hotx[ii],hoty[ii],:] = (did_clean[hotx[ii]-1,hoty[ii],:] + did_clean[hotx[ii]+1,hoty[ii],:] + did_clean[hotx[ii],hoty[ii]-1,:] + did_clean[hotx[ii],hoty[ii]+1,:])/4
# # did_clean_med = np.median(did_clean, axis=2)
# # for ii in range(did.shape[-1]):
# #     tmp = did_clean[:,:,ii]
# #     tmp[tmp > did_clean_med * 1.5] = np.nan
# #
# # plt.figure()
# # plt.imshow(np.nanmean(did_clean, axis=2), origin='lower')
# # plt.clim(200, 205)
# # plt.show()
# #
# # bgc = bg.copy()
# # bgc[bg > 250] = np.nan
# # bgc[..., 438] = np.nan
# # med = np.nanmedian(bg, axis=2)
# # avg = np.nanmean(bgc, axis=2)
# #
# # plt.figure()
# # plt.subplot(1, 2, 1)
# # plt.imshow(med, origin='lower')
# # plt.axis('off')
# # plt.clim(201, 203)
# # plt.plot(xy[:, 1], xy[:, 0], '.k')
# # plt.title('median \n black dots = max locations')
# # plt.colorbar()
# # plt.subplot(1, 2, 2)
# # plt.imshow(avg, origin='lower')
# # plt.axis('off')
# # plt.title('average\n')
# # plt.colorbar()
# # plt.clim(201, 203)
# # plt.show()
# #
# # medvec = np.nanmedian(np.nanmedian(bg, axis=0), axis=0)
# # plt.figure()
# # plt.plot((np.asarray(time)-59848)*24, medvec)
# # plt.show()
# #
# # didmed = np.median(did, axis=2)
# # didmed[didmed < 1] = 202
# # plt.figure()
# # plt.imshow(didmed)
# # plt.show()
# #
