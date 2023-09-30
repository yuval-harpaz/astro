from astro_utils import *
os.chdir('/media/innereye/My Passport/Data/JWST/HL-TAU/')

data = fits.open('jw01179-o015_t004_nircam_f405n-f444w-sub160p_i2d.fits')[1].data
adj = level_adjust(data, factor=1)
center = [[86,87], [92]]
for ii in center[0]:
    for jj in center[1]:
        adj[ii, jj] = 1

plt.imshow(adj, cmap='gray')
##
sketch = np.zeros((adj.shape[0], adj.shape[1], 3))
sketch = np.zeros((adj.shape[0], adj.shape[1], 3))
for ii in range(3):
    sketch[..., ii] = adj.copy()

clean = np.zeros(adj.shape)
for ph in range(center[0][-1], adj.shape[0]):
    for pw in range(center[1][-1], adj.shape[1]):
        point = [[ph, pw]]
        fliph = center[0][0] - (point[0][0] - center[0][-1])
        flipw = center[1][0] - (point[0][1] - center[1][-1])
        if fliph > -1 and flipw > -1:
            point.append([fliph, point[0][1]])
            point.append([point[0][0], flipw])
            point.append([fliph, flipw])
            point = np.array(point)
            val = adj[point[:, 0], point[:, 1]].copy()
            clean[point[:, 0], point[:, 1]] = val-np.min(val)

plt.imshow(clean, origin='lower')
##
path = ['jw01179-o014_t004_nircam_clear-f187n-sub160p_i2d.fits', 'jw01179-o015_t004_nircam_clear-f187n-sub160p_i2d.fits',
        'jw01179-o014_t004_nircam_clear-f200w-sub160p_i2d.fits', 'jw01179-o015_t004_nircam_clear-f200w-sub160p_i2d.fits',
        'jw01179-o014_t004_nircam_clear-f410m-sub160p_i2d.fits', 'jw01179-o015_t004_nircam_clear-f410m-sub160p_i2d.fits',
        'jw01179-o014_t004_nircam_f405n-f444w-sub160p_i2d.fits', 'jw01179-o015_t004_nircam_f405n-f444w-sub160p_i2d.fits']

 plt.figure()
 for ii in range(8):
     plt.subplot(2,4,ii+1)
     hdu = fits.open(path[ii])
     plt.imshow(level_adjust(hdu[1].data))

centers = [[[99, 100],[93, 94]],[[104, 105],[95, 96]],
           [[98],[92, 93]],[[102, 103],[94]],
           [[83, 84],[91, 92]],[[86, 87],[92, 93]],
           [[83, 84],[91, 92]],[[86],[92]]]


def cpsf(img, center):
    clean = np.zeros(img.shape)
    for ph in range(center[0][-1], img.shape[0]):
        for pw in range(center[1][-1], img.shape[1]):
            point = [[ph, pw]]
            fliph = center[0][0] - (point[0][0] - center[0][-1])
            flipw = center[1][0] - (point[0][1] - center[1][-1])
            if fliph > -1 and flipw > -1:
                point.append([fliph, point[0][1]])
                point.append([point[0][0], flipw])
                point.append([fliph, flipw])
                point = np.array(point)
                val = img[point[:, 0], point[:, 1]].copy()
                clean[point[:, 0], point[:, 1]] = val - np.min(val)
    return clean
##
 plt.figure()
 for ii in range(8):
     plt.subplot(2,4,ii+1)
     hdu = fits.open(path[ii])
     plt.imshow(level_adjust(cpsf(hdu[1].data, centers[ii])))

##
plt.figure()
for ii in range(8):
    plt.subplot(2, 4, ii + 1)
    hdu = fits.open(path[ii])
    plt.imshow(cpsf(level_adjust(hdu[1].data), centers[ii]))

for ii in range(len(path)):
    if ii == 0:
        hdu0 = fits.open(path[ii])
        img = hdu0[1].data
        img = hole_func_fill(img)
        img = cpsf(level_adjust(img), centers[ii])
        layers = np.zeros((img.shape[0], img.shape[1], len(path)))
        hdr0 = hdu0[1].header
        hdu0.close()
    else:
        hdu = fits.open(path[ii])
        hdu[1].data = hole_func_fill(hdu[1].data)
        hdu[1].data = cpsf(level_adjust(hdu[1].data), centers[ii])
        img, _ = reproject_interp(hdu[1], hdr0)
    layers[:, :, ii] = img


##
rgb = np.zeros((img.shape[0], img.shape[1], 3))
rgb[..., 0] = np.nanmin(layers[..., 4:6], 2)
rgb[..., 1] = np.nanmin(layers[..., 2:4], 2)
rgb[..., 2] = np.nanmin(layers[..., :2], 2)
rgb = blc_image(rgb)
# plt.imshow(rgb, origin='lower')
plt.imsave('min.png', rgb, origin='lower')

rgb = np.zeros((img.shape[0], img.shape[1], 3))
rgb[..., 0] = np.nanmean(layers[..., 4:6], 2)
rgb[..., 1] = np.nanmean(layers[..., 2:4], 2)
rgb[..., 2] = np.nanmean(layers[..., :2], 2)
rgb = blc_image(rgb)
# plt.imshow(rgb, origin='lower')
plt.imsave('mean.png', rgb, origin='lower')

##

for ii in range(len(path)):
    if ii == 0:
        hdu0 = fits.open(path[ii])
        img = hdu0[1].data
        img = hole_func_fill(img)
        img = level_adjust(img, factor=1)
        img = cpsf(img, centers[ii])
        layers = np.zeros((img.shape[0], img.shape[1], len(path)))
        hdr0 = hdu0[1].header
        hdu0.close()
    else:
        hdu = fits.open(path[ii])
        hdu[1].data = hole_func_fill(hdu[1].data)
        hdu[1].data = level_adjust(hdu[1].data, factor=1)
        hdu[1].data = cpsf(hdu[1].data, centers[ii])
        img, _ = reproject_interp(hdu[1], hdr0)
    layers[:, :, ii] = img

rgb = np.zeros((img.shape[0], img.shape[1], 3))
rgb[..., 0] = np.nanmin(layers[..., 4:6], 2)
rgb[..., 1] = np.nanmin(layers[..., 2:4], 2)
rgb[..., 2] = np.nanmin(layers[..., :2], 2)
rgb = blc_image(rgb)
# plt.imshow(rgb, origin='lower')
plt.imsave('min1.png', rgb, origin='lower')

rgb = np.zeros((img.shape[0], img.shape[1], 3))
rgb[..., 0] = np.nanmean(layers[..., 4:6], 2)
rgb[..., 1] = np.nanmean(layers[..., 2:4], 2)
rgb[..., 2] = np.nanmean(layers[..., :2], 2)
rgb = blc_image(rgb)
# plt.imshow(rgb, origin='lower')
plt.imsave('mean1.png', rgb, origin='lower')


##
for ii in range(len(path)):
    if ii == 0:
        hdu0 = fits.open(path[ii])
        img = hdu0[1].data
        img = hole_func_fill(img)
        img = cpsf(img, centers[ii])
        img = level_adjust(img, factor=4)
        layers = np.zeros((img.shape[0], img.shape[1], len(path)))
        hdr0 = hdu0[1].header
        hdu0.close()
    else:
        hdu = fits.open(path[ii])
        hdu[1].data = hole_func_fill(hdu[1].data)
        hdu[1].data = cpsf(hdu[1].data, centers[ii])
        hdu[1].data = level_adjust(hdu[1].data, factor=4)
        img, _ = reproject_interp(hdu[1], hdr0)
    layers[:, :, ii] = img

rgb = np.zeros((img.shape[0], img.shape[1], 3))
rgb[..., 0] = np.nanmin(layers[..., 4:6], 2)
rgb[..., 1] = np.nanmin(layers[..., 2:4], 2)
rgb[..., 2] = np.nanmin(layers[..., :2], 2)
rgb = blc_image(rgb)
# plt.imshow(rgb, origin='lower')
plt.imsave('min2.png', rgb, origin='lower')

rgb = np.zeros((img.shape[0], img.shape[1], 3))
rgb[..., 0] = np.nanmean(layers[..., 4:6], 2)
rgb[..., 1] = np.nanmean(layers[..., 2:4], 2)
rgb[..., 2] = np.nanmean(layers[..., :2], 2)
rgb = blc_image(rgb)
# plt.imshow(rgb, origin='lower')
plt.imsave('mean2.png', rgb, origin='lower')
