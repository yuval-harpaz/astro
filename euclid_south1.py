from astro_utils import *
os.chdir('/media/innereye/KINGSTON/Euclid/J041110.98-481939.3')
files = sorted(glob('*BGSUB*NIR*fits'))+glob('*BGSUB*VIS*fits')





for ii in range(len(files)):
    hdu = fits.open(files[ii])
    print(hdu[0].data.shape)
    hdu.close()

layers = np.zeros((2000, 2000, 3))
counter = -1
for ii in [0, 1, 3]:
    counter += 1
    hdu = fits.open(files[ii])
    layers[..., counter] = hole_func_fill(hdu[0].data[15000:17000, 8000:10000], fill_below=0)
    hdu.close()
    print(counter)

rgb = np.zeros_like(layers[..., :3])
for ii in range(3):
    rgb[..., ii] = level_adjust(log(layers[..., ii]), factor=3)
rgb = np.fliplr(transform.rotate(rgb, 90))
plt.figure()
plt.imshow(rgb)
plt.imsave('tmp3.jpg', rgb, pil_kwargs={'quality':95})
rgb = reduce_color(rgb, 2, replace=np.max)
rgb = reduce_color(rgb, 1, replace=np.max)
plt.imsave('tmp3_r2.jpg', rgb, pil_kwargs={'quality':95})



layers = np.zeros((2000, 2000, 4))
counter = -1
for ii in range(4):
    counter += 1
    hdu = fits.open(files[ii])
    layers[..., counter] = hole_func_fill(hdu[0].data[15000:17000, 8000:10000], fill_below=0)
    hdu.close()
    print(counter)

rgb = np.zeros_like(layers[..., :3])
for ii in range(3):
    rgb[..., ii] = level_adjust(log(layers[..., ii]), factor=3)
rgb = np.fliplr(transform.rotate(rgb, 90))
plt.figure()
plt.imshow(rgb)
plt.imsave('nisp3log.jpg', rgb, pil_kwargs={'quality':95})





fac = 3
rgb = np.zeros_like(layers[..., :3])
rgb[..., 0] = level_adjust(log(layers[..., 0]), factor=fac)
rgb[..., 2] = level_adjust(log(layers[..., 3]), factor=fac)
rgb[..., 1] = (level_adjust(log(layers[..., 1]), factor=fac) +
               level_adjust(log(layers[..., 2]), factor=fac))/2
rgb = np.fliplr(transform.rotate(rgb, 90))
rgb = reduce_color(rgb, 2, replace=np.max)
rgb = reduce_color(rgb, 1, replace=np.max)
plt.figure()
plt.imshow(rgb)
plt.imsave('euclid3log.jpg', rgb, pil_kwargs={'quality':95})
np.save('2kx2k.npy', layers)

## crop more
rgb = np.zeros_like(layers[..., :3])
rgb[..., 0] = level_adjust(log(layers[..., 0]), factor=fac)
rgb[..., 1] = level_adjust(log(layers[..., 1]), factor=fac)
rgb[..., 2] = (level_adjust(log(layers[..., 3]), factor=fac) +
               level_adjust(log(layers[..., 2]), factor=fac))/2
rgb = np.fliplr(transform.rotate(rgb, 90))
rgb = reduce_color(rgb, 2, replace=np.max)
rgb = reduce_color(rgb, 1, replace=np.max)
plt.figure()
plt.imshow(rgb)
plt.imsave('euclid3logB.jpg', rgb, pil_kwargs={'quality':95})

##
os.chdir('/media/innereye/KINGSTON/Euclid/South')
files = sorted(glob('*BGSUB*NIR*fits'))+glob('*BGSUB*VIS*fits')

size = 2000
start_h = 10000
start_w = 7500
tile = files[-1].split('TILE')[1].split('-')[0]
prefix = tile+'center_'
layers = np.zeros((size, size, 4))
counter = -1
for ii in range(4):
    counter += 1
    hdu = fits.open(files[ii])
    layers[..., counter] = hole_func_fill(hdu[0].data[start_h:start_h+size, start_w:start_w+size], fill_below=0)
    hdu.close()
    print(counter)

rgb = np.zeros_like(layers[..., :3])
for ii in range(3):
    rgb[..., ii] = level_adjust(log(layers[..., ii]), factor=3)
rgb = np.fliplr(transform.rotate(rgb, 90))
plt.figure()
plt.imshow(rgb)
plt.imsave(prefix+'nisp3log.jpg', rgb, pil_kwargs={'quality':95})





fac = 3
rgb = np.zeros_like(layers[..., :3])
rgb[..., 0] = level_adjust(log(layers[..., 0]), factor=fac)
rgb[..., 2] = level_adjust(log(layers[..., 3]), factor=fac)
rgb[..., 1] = (level_adjust(log(layers[..., 1]), factor=fac) +
               level_adjust(log(layers[..., 2]), factor=fac))/2
rgb = np.fliplr(transform.rotate(rgb, 90))
rgb = reduce_color(rgb, 2, replace=np.max)
rgb = reduce_color(rgb, 1, replace=np.max)
rgblog = rgb.copy()
plt.figure()
plt.imshow(rgblog)
plt.imsave(prefix+'euclid3log.jpg', rgblog, pil_kwargs={'quality':95})

fac = 3
rgb = np.zeros_like(layers[..., :3])
rgb[..., 0] = level_adjust(layers[..., 0], factor=fac)
rgb[..., 2] = level_adjust(layers[..., 3], factor=fac)
rgb[..., 1] = (level_adjust(layers[..., 1], factor=fac) +
               level_adjust(layers[..., 2], factor=fac))/2
rgb = np.fliplr(transform.rotate(rgb, 90))
rgb = reduce_color(rgb, 2, replace=np.max)
rgb = reduce_color(rgb, 1, replace=np.max)
plt.figure()
plt.imshow(rgb)
plt.imsave(prefix+'euclid3.jpg', rgb, pil_kwargs={'quality':95})


rgbhalflog = (rgb + rgblog)/2
plt.figure()
plt.imshow(rgbhalflog)
plt.imsave(prefix+'euclid3halflog.jpg', rgbhalflog, pil_kwargs={'quality':95})
