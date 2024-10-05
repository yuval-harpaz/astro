from astro_utils import *

pat = '/media/innereye/KINGSTON/JWST/NGC891/'
os.chdir(pat)
files = glob('*.fits')
filt = filt_num(files)
files = np.array(files)[np.argsort(filt)]
xy, size = mosaic_xy(files, plot=False)
layers = mosaic(files, xy=xy, size=size, method='layers')
data = np.zeros((2110, 1777, 3))
for ii in range(3):
    data[..., ii] = np.nanmean(layers[..., ii*4:ii*4+4], 2)
img = np.zeros((2110, 1777, 3))
for ii in range(3):
    img[..., 2-ii] = level_adjust(data[..., ii])
plt.imshow(img, origin='lower')
plt.imsave('fac4.jpg', img, origin='lower', pil_kwargs={'quality':95})

# img = np.zeros((2110, 1777, 3))
# for ii in range(3):
#     img[..., 2-ii] = level_adjust(data[..., ii], factor=1)
# plt.imshow(img, origin='lower')
# plt.imsave('fac1.jpg', img, origin='lower', pil_kwargs={'quality':95})
##
filt = [2100, 1130, 770]
col = matplotlib.cm.jet(filt / np.max(filt))[:, :3]  # [:, ::-1]
rgb = assign_colors(data, col)
for ic in range(3):
    rgb[:, :, ic] = rgb[:, :, ic] * 255
rgb = rgb.astype('uint8')
rgb = blc_image(rgb)
plt.imsave('filt4.jpg', img, origin='lower', pil_kwargs={'quality':95})