from astro_utils import *
os.chdir('/media/innereye/My Passport/Data/JWST/belt/')

path = glob('*long*.fits')
print(path)

long = reproject(path)
long_mean = np.zeros(long.shape)
for ii in range(len(path)):
    long_mean[..., ii] = level_adjust(long[..., ii])
long_mean[long_mean == 0] = np.nan
long_mean = np.nanmedian(long_mean, 2)
plt.figure()
plt.imshow(long_mean, origin='lower', cmap='gray')
##
plt.figure()
for ii in range(len(path)):
    plt.subplot(2,2, ii+1)
    hdu = fits.open(path[ii])
    plt.imshow(hdu[1].data[500:900, 1700:])
    plt.clim(1, 5)
##
med = np.nanmedian(long[..., [0, 1, 3]], 2)
xy = hole_xy(med)
size = hole_size(med, xy, plot=False)
filled = hole_disk_fill(med, xy, size, larger_than=0)
filled1 = hole_conv_fill(filled)
cmap = 'gray'
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(med, origin='lower', cmap=cmap)
plt.clim(1, 7)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(med[650:800, 1900:], origin='lower', cmap=cmap)
plt.clim(1, 7)
plt.axis('off')
##
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(filled1, origin='lower', cmap=cmap)
plt.clim(1, 7)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(filled1[650:800, 1900:], origin='lower', cmap=cmap)
plt.clim(1, 7)
plt.axis('off')