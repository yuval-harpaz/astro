%matplotlib qt
from astro_utils import *
from astro_fill_holes import *
import os
import pandas as pd
from glob import glob
import numpy as np

os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/')
stuff = '/home/innereye/astro/dart/'
path = np.asarray(list_files('/home/innereye/Dropbox/Moris_20220926-20221002/', '*.fits'))
kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data
mmdd = np.asarray([x[9:13] for x in path])
mmddu = np.unique(mmdd)
df = pd.read_csv(stuff+'meta.csv')
## get center of didymos for al images
if False:
    file = df['file'].to_numpy()
    for ff in file:
        hdu = fits.open(ff)
        img = hdu[0].data.copy()
        img[img > 205] = 210
        img[img < 198] = 198
        plt.imsave(stuff+'png/'+ff[:-4]+'png', img,origin='lower')
        print(ff)

##
halfdid = 500
bad = glob(stuff+'png/bad/*.png')
bad = [x[x.find('mor'):] for x in bad]
bad = [x.replace('.png','.fits') for x in bad]
isgood = np.ones(len(df), bool)
for ii in range(len(isgood)):
    if df['file'][ii][14:] in np.asarray(bad):
        isgood[ii] = False
filt = np.asarray([x[-1].lower() for x in df['filter']])
df = df[(filt == 'v') & (df['x'] > 0) & isgood]
mmdd = np.asarray([x[9:13] for x in df['file']])
data = np.zeros((halfdid*2+1, halfdid*2+1, 7))
raw = data.copy()
# ops = np.zeros((halfdid*2, halfdid*2, 7), 'uint8')
nonan = np.zeros(7)
for day in range(7):
    date = mmddu[day]
    print(date)
    pathc = df['file'].to_numpy()
    pathc = pathc[mmdd == date]
    x = df['x'].to_numpy()[mmdd == date]
    y = df['y'].to_numpy()[mmdd == date]
    n = len(pathc)
    print(n)
    cube = np.zeros((halfdid*2+1, halfdid*2+1, n))
    cube[...] = np.nan
    for ii in range(n):
        # print(ii)
        f = pathc[ii]
        hdu = fits.open(f)
        # n += 1
        img = hdu[0].data.copy()
        img = img.astype(float)
        img[:,:26] = np.nan
        img[484:,:] = np.nan
        x0 = halfdid + 1 - x[ii]
        y0 = halfdid + 1 - y[ii]
        tmp = np.zeros((halfdid*2+1,halfdid*2+1))
        tmp[...] = np.nan
        tmp[x0:x0+512,y0:y0+512] = img
        cube[:,:,ii] = tmp
    med = np.zeros((halfdid*2+1,halfdid*2+1))
    med[:,:] = np.nan
    for jj in range(med.shape[0]):
        for kk in range(med.shape[1]):
            med[jj, kk] = np.nanmedian(cube[jj, kk,:])
    # med = np.nanmedian(cube, axis=2)
    for jj in range(n):
        tmp = cube[:,:,jj]
        tmp[tmp > med*1.5] = np.nan
        cube[:,:,jj] = tmp
    # ignore smeared asteroid
    # noise = cube.copy()
    ns = np.zeros(cube.shape[2])
    for ii in range(cube.shape[2]):
        noise = cube[:,:,ii].copy()
        noise[halfdid - 30:halfdid + 31, halfdid - 30:halfdid + 31] = np.nan
        ns[ii] = np.sum(noise[:,:] > 210)
        if ns[ii] > 15:
            cube[:,:,ii] = np.nan
        else:
            cube[:,:,ii] = cube[:,:,ii] - np.nanmedian(noise)
    nonan[day] = np.sum(~np.isnan(cube[halfdid,halfdid,:]))
    d = np.zeros((halfdid*2+1,halfdid*2+1))
    for jj in range(med.shape[0]):
        for kk in range(med.shape[1]):
            d[jj, kk] = np.nanmean(cube[jj, kk,:])
    del cube
    raw[:, :, day] = d.copy()
    nans = np.isnan(d)
    d[nans] = np.nanpercentile(d[630:, :], 50)
    badx = np.where(np.std(d,0) > 0.5)[0]
    badx = badx[(badx < 450) | (badx > 550)]
    bady = np.where(np.std(d,0) > 0.5)[0]
    bady = bady[(bady < 450) | (bady > 550)]
    d[badx,:] = np.nan
    d[:, bady] = np.nan
    # print('badx')
    # print(badx)
    d = convolve(d, kernel=kernel.array)
    d[nans] = np.nan
    data[:, :, day] = d

# ok = np.where(np.sum(np.isnan(data[:,halfdid+1,1]),1) == 0)[0]
ok = np.where(~np.isnan(data[:,halfdid+1,1]))[0]
cropped = data[ok[0]:ok[-1]+1,:,:]
ok = np.where(~np.isnan(data[halfdid+1,:,1]))[0]
cropped = cropped[:,ok[0]:ok[-1]+1,:]
cropped[96:515,:,ii]
##
plt.figure()
for ii in range(7):
    prc = np.nanpercentile(cropped[:,:,ii], 25)
    plt.subplot(2,4,ii+1)
    plt.imshow(cropped[96:515,:,ii]-prc)
    plt.clim(0, 1)

##
with open(stuff+'per_day_smooth2.pkl', 'wb') as f:
    pickle.dump(cropped, f)

ok = np.where(~np.isnan(data[:,halfdid+1,1]))[0]
cropped = raw[ok[0]:ok[-1]+1,:,:]
ok = np.where(~np.isnan(data[halfdid+1,:,1]))[0]
cropped = cropped[:,ok[0]:ok[-1]+1,:]
cropped[96:515,:,ii]

with open(stuff+'per_day_raw2.pkl', 'wb') as f:
    pickle.dump(cropped, f)

# out = cv2.VideoWriter(stuff+'per_day.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 3, (300, 300), False)
# for day in range(7):
#     out.write(ops[:,:,day])
# out.release()
#

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
