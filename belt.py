import matplotlib.pyplot as plt

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
##
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(level_adjust(filled1, factor=1), origin='lower', cmap=cmap)
# plt.clim(1, 7)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(level_adjust(filled1[650:800, 1900:], factor=1), origin='lower', cmap=cmap)
plt.clim(0.15, 1)
plt.axis('off')
## not long
path = glob('*.fits')
path = [x for x in path if not 'long' in x]
print(path)
##
plt.figure()
for ii in range(len(path)):
    plt.subplot(2,3, ii+1)
    hdu = fits.open(path[ii])
    plt.imshow(hdu[1].data)
    plt.clim(1, 5)
    plt.title(hdu[0].header['FILTER'])
##
path = glob('*.fits')
path = [x for x in path if not 'long' in x]
path = glob('*long*.fits') + path
layers = reproject(path)


long = layers[..., :4]
long[long == 0] = np.nan
med = np.nanmedian(long, 2)
xy = hole_xy(med)
size = hole_size(med, xy, plot=False)
filled = hole_disk_fill(med, xy, size, larger_than=0)
filled1 = hole_conv_fill(filled)

short = layers[..., 4:]
short[short == 0] = np.nan
med = np.nanmedian(short, 2)
# xy = hole_xy(med)
# size = hole_size(med, xy, plot=False)
# filled = hole_disk_fill(med, xy, size, larger_than=0)
filled2 = med.copy()
filled2[:1000, 1000:] = hole_conv_fill(med[:1000, 1000:])
f0 = filled1[:1000, 1000:]/5*255
f0[f0 < 0] = 0
f0[f0 > 255] = 255
f2 = filled2[:1000, 1000:]/5*255
f2[f2 < 0] = 0
f2[f2 > 255] = 255
f1 = (f0 + f2) / 2
rgb = np.zeros((1000, 1048, 3)).astype('uint8')
rgb[..., 0] = f0
rgb[..., 1] = f1
rgb[..., 2] = f2
plt.figure()
plt.imshow(rgb, origin='lower')
plt.imsave('277_mean_200.png', rgb, origin='lower')


layers[layers == 0] = np.nan
med = np.nanmedian(layers[:1000, 1000:, :], 2)
filled3 = med.copy()
filled3 = hole_conv_fill(med)
f0 = filled3[:1000, 1000:]/5*255
f0[f0 < 0] = 0
f0[f0 > 255] = 255
plt.figure()
plt.imshow(f0, origin='lower')
plt.imsave('277_mean_200.png', rgb, origin='lower')
plt.imshow(med, cmap='gray', origin='lower'); plt.clim(1, 5)
# long_mean = np.zeros(long.shape)
# for ii in range(len(path)):
#     long_mean[..., ii] = level_adjust(long[..., ii])
# long_mean[long_mean == 0] = np.nan
# long_mean = np.nanmedian(long_mean, 2)
# plt.figure()
# plt.imshow(long_mean, origin='lower', cmap='gray')
# ##
# plt.figure()
# for ii in range(len(path)):
#     plt.subplot(2,2, ii+1)
#     hdu = fits.open(path[ii])
#     plt.imshow(hdu[1].data[500:900, 1700:])
#     plt.clim(1, 5)
##
med = np.nanmedian(long[..., [0, 1, 3]], 2)
xy = hole_xy(med)
size = hole_size(med, xy, plot=False)
filled = hole_disk_fill(med, xy, size, larger_than=0)
filled1 = hole_conv_fill(filled)
cmap = 'gray'

#
# long = reproject(path)
# long_mean = np.zeros(long.shape)
# for ii in range(len(path)):
#     long_mean[..., ii] = level_adjust(long[..., ii])
# long_mean[long_mean == 0] = np.nan
# long_mean = np.nanmedian(long_mean, 2)
# plt.figure()
# plt.imshow(long_mean, origin='lower', cmap='gray')
##
# plt.figure()
# for ii in range(len(path)):
#     plt.subplot(2,2, ii+1)
#     hdu = fits.open(path[ii])
#     plt.imshow(hdu[1].data[500:900, 1700:])
#     plt.clim(1, 5)

