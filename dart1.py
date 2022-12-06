from astro_utils import *
from astro_fill_holes import *
import os

## Take the first day data (before impact) to compute flat field
os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/Moris20220926/')
path = list_files('/home/innereye/Dropbox/Moris_20220926-20221002/Moris20220926/', '*.fits')
kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data
bg = np.zeros((496, 487, len(path)), 'float')
xy = np.zeros((len(path), 2), int)
halfmask = 30
filt = []
time = []
for ii in range(len(path)):
    f = path[ii]
    hdu = fits.open(f)
    img = hdu[0].data.copy()
    img = img.astype(float)
    img = img[:496, 25:]
    # smooth data before looking for max to avoid stars / noise
    smoothed = convolve(img, kernel=kernel.array)
    xyc = np.unravel_index(smoothed.argmax(), smoothed.shape)  # current x y coordinates of asteroid
    xy[ii, :] = [xyc[0], xyc[1]]
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

medvec = np.nanmedian(np.nanmedian(bg,axis=0),axis=0)
plt.figure()
plt.plot((np.asarray(time)-59848)*24,medvec)
plt.show()