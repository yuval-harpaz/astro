from astro_utils import *

pat = '/media/innereye/KINGSTON/JWST/data/NGC-1333/'
os.chdir(pat)
files = glob('*.fits')
# path = log['file']
img = np.zeros((6898, 6844, 3))
for ii in [0, 1]:
    hdu = fits.open(files[ii])
    if ii == 0:
        hdr0 = hdu[1].header
    lay = hdu[1].data
    lay = hole_func_fill(lay)
    lay = deband_layer(lay, flip=True, func=np.percentile)
    if ii == 1:
        hdu[1].data = lay
        lay, _ = reproject_interp(hdu[1], hdr0)
    img[..., 2 - 2 * ii] = lay
np.save('data.npy', img)
for ii in [0, 2]:
    img[..., ii] = level_adjust(img[..., ii], factor=1)
img[..., 1] = (img[..., 0] + img[..., 2])/2
plt.imsave('deband2.jpg', img, origin='lower', pil_kwargs={'quality':95})
##
# hdu0[1].data = deband_layer(hdu0[1].data, flip=True, func=np.nanpercentile)
img[..., 0] = level_adjust(hdu0[1].data, factor=1)
hdu = fits.open(files[1])
hdu[1].data = deband_layer(hdu[1].data, flip=True, func=np.nanpercentile)
img[..., 2], _ = reproject_interp(hdu[1], )
img[..., 1] = (img[..., 0] + img[..., 2])/2
plt.imsave('deband.png', img, origin='lower')
##
for ii in [0, 2]:
    img[..., ii] = deband_layer(img[..., ii], flip=True)
img[..., 1] = (img[..., 0] + img[..., 2])/2
plt.imsave('deband.png', img)
##
mn = np.min(img[..., np.array([0,2])], 2)
for ii in [0, 2]:
    noise = (img[..., ii] > 0.95) & ((img[..., ii] - img[..., 2-ii]) > 0.3)
    im = img[..., ii]
    im[noise] = mn[noise]
img[..., 1] = (img[..., 0] + img[..., 2])/2

# img1 = level_adjust(hdu[1].data[:-1, :-1], factor=1)
##
# img[..., 2] = np.roll(img1.copy(), (-20, 20))
plt.figure()
plt.imshow(img, origin='lower')
