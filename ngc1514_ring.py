"""Crystal ball nebula."""
from astro_utils import *
from astropy.convolution import Ring2DKernel
os.chdir('/media/innereye/KINGSTON/JWST/data/')

ring = Ring2DKernel(31, 3)
fm = glob('NGC1514/*min*.fits')
for ii in range(3):
    h = fits.open(fm[ii])
    med = median_filter(h[1].data, footprint=ring.array)
    med[med == 0] = h[1].data[med == 0]
    med[h[1].data <= 0] = 0
    h[1].data[h[1].data > med] = med[h[1].data > med]  # np.nanmin([med, h[1].data], axis=0)
    # h[1].data[h[1].data < 0] = 0
    h.writeto(fm[ii].replace('min','ring2'))


files = sorted(glob('NGC-1514*/*.fits'))
filt = filt_num(files)
files = np.array(files)[np.argsort(filt)]
files = [x for x in files if 'IMAGE-BKGNG' not in x]
filt = [2550, 1280, 770]
filts = ['f770w', 'f1280w', 'f2550w']

folders = np.unique([f.split('/')[0] for f in files])
nfolders = len(folders)
##
bck_dict = dict(zip(filts, glob('NGC1514/min*.fits')))
xy, size = mosaic_xy(files, plot=False)
layers = mosaic(files, xy=xy, size=size, method='layers', fill=False, subtract=bck_dict)
layers[layers == 0] = np.nan
data = np.zeros((layers.shape[0], layers.shape[1], 3))
for ii in range(3):
    data[..., ii] = np.nanmax(layers[..., ii*nfolders:ii*nfolders+nfolders], 2)
data[np.isnan(data)] = 0
datan = data.copy()
img = np.zeros((datan.shape[0], datan.shape[1], 3))
for ii in range(3):
    img[..., 2-ii] = level_adjust(datan[..., ii])
img = reduce_color(img, 2, thratio=2)
# plt.figure()
# plt.imshow(img)
plt.imsave('NGC1514/rgb_ring2_max.png', img, origin='lower')
col = matplotlib.cm.jet(filt / np.max(filt))[:, :3]  # [:, ::-1]
rgb = assign_colors(img, col)
for ic in range(3):
    rgb[:, :, ic] = rgb[:, :, ic] * 255
rgb = rgb.astype('uint8')
rgb = blc_image(rgb)
plt.figure()
plt.imshow(rgb, origin='lower')
plt.imsave('NGC1514/filt_ring2_max2.jpg', rgb, origin='lower', pil_kwargs={'quality': 95})
# ##
# layers = mosaic(files, xy=xy, size=size, method='layers', fill=False, subtract=bck_dict)
# layers[layers == 0] = np.nan
# data = np.zeros((layers.shape[0], layers.shape[1], 3))
# for ii in range(3):
#     data[..., ii] = np.nanmedian(layers[..., ii*nfolders:ii*nfolders+nfolders], 2)
# datan = data.copy()
# img = np.zeros((datan.shape[0], datan.shape[1], 3))
# for ii in range(3):
#     img[..., 2-ii] = level_adjust(datan[..., ii])
# filt = [2550, 1280, 770]
# col = matplotlib.cm.jet(filt / np.max(filt))[:, :3]  # [:, ::-1]
# rgb = assign_colors(img, col)
# for ic in range(3):
#     rgb[:, :, ic] = rgb[:, :, ic] * 255
# rgb = rgb.astype('uint8')
# rgb = blc_image(rgb)
# plt.imsave('/home/innereye/Pictures/ngc1514nofill.jpg', rgb, origin='lower', pil_kwargs={'quality': 95})
