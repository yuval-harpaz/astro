from astro_utils import *
os.chdir('/media/innereye/KINGSTON/Euclid/NGC6543')
files = sorted(glob('*NIR*fits'))+glob('*VIS*fits')
for ii in range(len(files)):
    hdu = fits.open(files[ii])
    print(hdu[0].data.shape)
    hdu.close()

layers = np.zeros((5000, 5000, 4))
for ii in range(4):
    hdu = fits.open(files[ii])
    layers[..., ii] = hole_func_fill(hdu[0].data[12000:17000, 1500:6500], fill_below=0)
    hdu.close()
    print(ii)

rgb = np.zeros_like(layers[..., :3])
for ii in range(3):
    rgb[..., ii] = level_adjust(layers[..., ii], factor=1)
rgb = np.fliplr(transform.rotate(rgb, 90))
plt.figure()
plt.imshow(rgb)
plt.imsave('nisp1.jpg', rgb, pil_kwargs={'quality':95})

rgb = np.zeros_like(layers[..., :3])
for ii in range(3):
    rgb[..., ii] = level_adjust(log(layers[..., ii]), factor=1)
rgb = np.fliplr(transform.rotate(rgb, 90))
plt.figure()
plt.imshow(rgb)
plt.imsave('nisp1log.jpg', rgb, pil_kwargs={'quality':95})

rgb = np.zeros_like(layers[..., :3])
rgb[..., 0] = level_adjust(log(layers[..., 0]), factor=1)
rgb[..., 2] = level_adjust(log(layers[..., 3]), factor=1)
rgb[..., 1] = (level_adjust(log(layers[..., 1]), factor=1) +
               level_adjust(log(layers[..., 2]), factor=1))/2
rgb = np.fliplr(transform.rotate(rgb, 90))
plt.figure()
plt.imshow(rgb)
plt.imsave('euclid1log.jpg', rgb, pil_kwargs={'quality':95})
np.save('5kx5k.npy', layers)

## crop more
rgb = np.zeros((510, 460, 3))
for ii in range(3):
    rgb[..., ii] = level_adjust(layers[2130:2640, 1820:2280, ii], factor=1)
rgb = np.fliplr(transform.rotate(rgb, 90, resize=True))
plt.figure()
plt.imshow(rgb)
plt.imsave('nisp1_crop.jpg', rgb, pil_kwargs={'quality':95})

rgb = np.zeros((510, 460, 3))
rgb[..., 0] = level_adjust(layers[2130:2640, 1820:2280, 0], factor=1)
rgb[..., 2] = level_adjust(layers[2130:2640, 1820:2280, 3], factor=1)
rgb[..., 1] = (level_adjust(layers[2130:2640, 1820:2280, 1], factor=1) +
               level_adjust(layers[2130:2640, 1820:2280, 2], factor=1))/2
rgb = np.fliplr(transform.rotate(rgb, 90, resize=True))
plt.figure()
plt.imshow(rgb)
plt.imsave('euclid1_crop.jpg', rgb, pil_kwargs={'quality':95})
rgb = np.zeros((510, 460, 3))
for ii in range(3):
    rgb[..., ii] = level_adjust(log(layers[2130:2640, 1820:2280, ii]), factor=1)
rgb = np.fliplr(transform.rotate(rgb, 90, resize=True))
plt.figure()
plt.imshow(rgb)
plt.imsave('nisp1log_crop.jpg', rgb, pil_kwargs={'quality':95})

rgb = np.zeros((510, 460, 3))
rgb[..., 0] = level_adjust(log(layers[2130:2640, 1820:2280, 0]), factor=1)
rgb[..., 2] = level_adjust(log(layers[2130:2640, 1820:2280, 3]), factor=1)
rgb[..., 1] = (level_adjust(log(layers[2130:2640, 1820:2280, 1]), factor=1) +
               level_adjust(log(layers[2130:2640, 1820:2280, 2]), factor=1))/2
rgb = np.fliplr(transform.rotate(rgb, 90, resize=True))
plt.figure()
plt.imshow(rgb)
plt.imsave('euclid1log_crop.jpg', rgb, pil_kwargs={'quality':95})
np.save('crop.npy', layers)

