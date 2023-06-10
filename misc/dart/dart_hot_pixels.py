import matplotlib.pyplot as plt
import numpy as np
%matplotlib qt
from astro_utils import *
from astro_fill_holes import *
import os
import pandas as pd
## Take the first day data (before impact) to compute flat field
os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/Moris20220926/')
stuff = '/home/innereye/astro/dart/'
path = list_files('/home/innereye/Dropbox/Moris_20220926-20221002/Moris20220926/', '*.fits')
kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data

## get center of didymos for 29 Nov
xy = np.zeros((len(path), 2), int)
for ii in range(len(path)):
    f = path[ii]
    hdu = fits.open(f)
    img = hdu[0].data.copy()
    img = img.astype(float)
    # img = img[:496, 25:]
    # smooth data before looking for max to avoid stars / noise
    smoothed = convolve(img, kernel=kernel.array)
    xyc = np.unravel_index(smoothed.argmax(), smoothed.shape)  # current x y coordinates of asteroid
    xy[ii, :] = [xyc[0], xyc[1]]
    print(ii)
np.savetxt(stuff + 'xy0926.csv', xy, delimiter=',', fmt='%d')

##  Get indices of hot pixels
xy = np.genfromtxt(stuff+'xy0926.csv', delimiter=',').astype(int)  # created with dart1

halfmask = 30
halfdid = 150
bg = np.zeros((512, 512, len(path)), 'float')  # Background
did = np.zeros((halfdid*2,halfdid*2, len(path)), 'float')  # Didymos
filt = []
time = []
for ii in range(len(path)):
    f = path[ii]
    hdu = fits.open(f)
    img = hdu[0].data.copy()
    img = img.astype(float)
    # img = img[:496, 25:]
    did[:, :, ii] = img[xy[ii,0]-halfdid:xy[ii,0]+halfdid,xy[ii,1]-halfdid:xy[ii,1]+halfdid]
    # smooth data before looking for max to avoid stars / noise
    ## mask a square around asteroid
    img[xy[ii, 0]-halfmask:xy[ii, 0]+halfmask, xy[ii, 1]-halfmask:xy[ii, 1]+halfmask] = np.nan
    bg[..., ii] = img
    time.append(hdu[0].header['MJD_OBS'])
    filt.append(hdu[0].header['FILTER'])
    print(ii)

hotmed = np.nanmedian(bg, axis=2)
hotm = np.nanmean(bg, axis=2)
hot = hotmed  > 202
hotx, hoty = np.where(hot)
for ii in range(len(hotx)):
    hotv = hotm[hotx, hoty]

df = pd.DataFrame(hotx,columns=['x'])
df['y'] = hoty
df['mean'] = hotv
df = df.sort_values('mean', ascending=False)
df.to_csv(stuff+'hot_pixels.csv',index=False)

## create didymos and background after fixing hot pixels and stars
df = pd.read_csv(stuff+'hot_pixels.csv')
did_clean = np.zeros((300, 300, len(path)))
bg_clean = np.zeros((512, 512, len(path)))
# fix hot pixels
for ii in range(len(path)):
    f = path[ii]
    hdu = fits.open(f)
    img = hdu[0].data.copy()
    img = img.astype(float)
    for pix in range(3):
        x = df['x'].loc[pix]
        y = df['y'].loc[pix]
        img[x,y] = (img[x-1,y] + img[x+1,y] + img[x, y-1] + img[x, y+1])/4
    # img = img[:496, 25:]
    did_clean[:, :, ii] = img[xy[ii,0]-halfdid:xy[ii,0]+halfdid,xy[ii,1]-halfdid:xy[ii,1]+halfdid]
    # smooth data before looking for max to avoid stars / noise
    ## mask a square around asteroid
    img[xy[ii, 0]-halfmask:xy[ii, 0]+halfmask, xy[ii, 1]-halfmask:xy[ii, 1]+halfmask] = np.nan
    bg_clean[..., ii] = img

did_clean_med = np.median(did_clean, axis=2)
for ii in range(did.shape[-1]):
    tmp = did_clean[:,:,ii]
    tmp[tmp > did_clean_med * 1.5] = np.nan
did_mean = np.nanmean(did_clean, axis=2)
plt.figure()
plt.imshow(did_mean, origin='lower')
plt.clim(200, 205)
plt.show()
np.savetxt(stuff + 'Didymos0926.csv', did_mean, delimiter=',', fmt='%1.6f')

bg_clean_med = np.nanmedian(bg_clean, axis=2)
bg_cleanless = bg_clean.copy()
for ii in range(did.shape[-1]):
    tmp = bg_cleanless[:,:,ii]
    tmp[tmp > bg_clean_med * 1.5] = np.nan

bg_mean = np.nanmean(bg_cleanless, axis=2)
np.savetxt(stuff + 'flat_field.csv', bg_mean, delimiter=',', fmt='%1.6f')

#
# plt.figure()
# for ii in range(len(hotx)):
#     plt.subplot(4, 4, ii+1)
#     plt.plot(bg[hotx[ii]+1, hoty[ii], :])
#     plt.plot(bg[hotx[ii], hoty[ii], :])
#     plt.ylim(195, 210)
#     plt.plot([0,467],[202,202],'k')
#     plt.title(str(hotx[ii])+','+str(hoty[ii]))
#     plt.xticks([])
#     if ii == 0:
#         plt.legend(['control pixel','hot pixel','median'])
# plt.suptitle('hot pixels behavior for 26 Nov data')
# plt.show()
#
# plt.figure()
# plt.imshow(np.mean(did, axis=2), origin='lower')
# plt.clim(200, 205)
# plt.show()
#
#
# did_clean = did.copy()
# for ii in range(3):
#     did_clean[hotx[ii],hoty[ii],:] = (did_clean[hotx[ii]-1,hoty[ii],:] + did_clean[hotx[ii]+1,hoty[ii],:] + did_clean[hotx[ii],hoty[ii]-1,:] + did_clean[hotx[ii],hoty[ii]+1,:])/4
# did_clean_med = np.median(did_clean, axis=2)
# for ii in range(did.shape[-1]):
#     tmp = did_clean[:,:,ii]
#     tmp[tmp > did_clean_med * 1.5] = np.nan
#
# plt.figure()
# plt.imshow(np.nanmean(did_clean, axis=2), origin='lower')
# plt.clim(200, 205)
# plt.show()
#
# bgc = bg.copy()
# bgc[bg > 250] = np.nan
# bgc[..., 438] = np.nan
# med = np.nanmedian(bg, axis=2)
# avg = np.nanmean(bgc, axis=2)
#
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(med, origin='lower')
# plt.axis('off')
# plt.clim(201, 203)
# plt.plot(xy[:, 1], xy[:, 0], '.k')
# plt.title('median \n black dots = max locations')
# plt.colorbar()
# plt.subplot(1, 2, 2)
# plt.imshow(avg, origin='lower')
# plt.axis('off')
# plt.title('average\n')
# plt.colorbar()
# plt.clim(201, 203)
# plt.show()
#
# medvec = np.nanmedian(np.nanmedian(bg, axis=0), axis=0)
# plt.figure()
# plt.plot((np.asarray(time)-59848)*24, medvec)
# plt.show()
#
# didmed = np.median(did, axis=2)
# didmed[didmed < 1] = 202
# plt.figure()
# plt.imshow(didmed)
# plt.show()
#
