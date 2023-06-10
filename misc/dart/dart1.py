from astro_utils import *
from astro_fill_holes import *
import os
# import pandas as pd
## Take the first day data (before impact) to compute flat field
os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/Moris20220926/')
stuff = '/home/innereye/Data/DART/stuff/'
path = list_files('/home/innereye/Dropbox/Moris_20220926-20221002/Moris20220926/', '*.fits')
kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data

xy = np.zeros((len(path), 2), int)
#  Find center of Didymos
# if os.path.isfile(stuff+'xy26.csv'):
# xy = pd.read_csv(stuff+'xy26.csv')
# xy = np.genfromtxt(stuff+'xy26.csv', delimiter=',').astype(int)
# else:
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
np.savetxt(stuff + 'xy26.csv', xy, delimiter=',', fmt='%d')
halfmask = 30
halfdid = 150
bg = np.zeros((496, 487, len(path)), 'float')  # Background
did = bg.copy()  # Didymos
filt = []
time = []
for ii in range(len(path)):
    f = path[ii]
    hdu = fits.open(f)
    img = hdu[0].data.copy()
    img = img.astype(float)
    img = img[:496, 25:]
    did[248-halfdid:248+halfdid, 244-halfdid:244+halfdid, ii] = img[xy[ii,0]-halfdid:xy[ii,0]+halfdid,xy[ii,1]-halfdid:xy[ii,1]+halfdid]
    # smooth data before looking for max to avoid stars / noise
    ## mask a square around asteroid
    img[xyc[0]-halfmask:xyc[0]+halfmask, xyc[1]-halfmask:xyc[1]+halfmask] = np.nan
    bg[..., ii] = img
    time.append(hdu[0].header['MJD_OBS'])
    filt.append(hdu[0].header['FILTER'])
    print(ii)


bgc = bg.copy()
bgc[bg > 250] = np.nan
bgc[..., 438] = np.nan
med = np.nanmedian(bg, axis=2)
avg = np.nanmean(bgc, axis=2)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(med, origin='lower')
plt.axis('off')
plt.clim(201, 203)
plt.plot(xy[:, 1], xy[:, 0], '.k')
plt.title('median \n black dots = max locations')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(avg, origin='lower')
plt.axis('off')
plt.title('average\n')
plt.colorbar()
plt.clim(201, 203)
plt.show()

medvec = np.nanmedian(np.nanmedian(bg, axis=0), axis=0)
plt.figure()
plt.plot((np.asarray(time)-59848)*24, medvec)
plt.show()

didmed = np.median(did, axis=2)
didmed[didmed < 1] = 202
plt.figure()
plt.imshow(didmed)
plt.show()

plt.figure()
plt.imshow(np.mean(did, axis=2), origin='lower')
plt.clim(200, 205)
plt.show()