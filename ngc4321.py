from astro_utils import *

log = pd.read_csv('logs/NGC-4321.csv')
def crval_fix1(hd):
    logrow = np.where(log['file'] == path[ii])[0]
    if len(logrow) == 1:
        for cr in [1, 2]:
            correct = log.iloc[logrow][f'CRVAL{cr}fix'].to_numpy()[0]
            if ~np.isnan(correct):
                hd[1].header[f'CRVAL{cr}'] = correct
    return hd
pat = '/media/innereye/My Passport/Data/JWST/data/NGC-4321/'
path = log['file']
filt = filt_num(path)
path = np.array(path)[np.argsort(filt)]
ii = 8
hdu0 = fits.open(pat+path[ii])
hdu0 = crval_fix1(hdu0)
img = hdu0[1].data
layers = np.zeros((img.shape[0], img.shape[1], len(path)))
layers[:, :, ii] = img
hdr0 = hdu0[1].header
hdu0.close()
for ii in range(len(path)):
    if ii == 8:
        hdu = fits.open(pat + path[ii])
    else:
        hdu = fits.open(pat+path[ii])
        hdu = crval_fix1(hdu)
    hdu[1].data = hole_func_fill(hdu[1].data)
    img, _ = reproject_interp(hdu[1], hdr0)
    layers[..., ii] = level_adjust(img)


# for ii in range(len(path)):
#     layers[..., ii] = hole_func_fill(layers[..., ii])
#     layers[..., ii] = level_adjust(layers[..., ii])

lay8 = np.zeros((img.shape[0], img.shape[1], 8))
lay8[..., 0] = np.nanmean(layers[..., 0:2], 2)
lay8[..., 1] = np.nanmean(layers[..., 2:4], 2)
lay8[..., 2] = np.nanmean(layers[..., 4:6], 2)
lay8[..., 3] = np.nanmean(layers[..., 6:8], 2)
for ii in range(4):
    lay8[..., ii+4] = layers[..., ii+8]

for ii in range(8):
    plt.subplot(2,4, ii + 1)
    plt.imshow(lay8[..., ii], cmap='gray')
    plt.axis('off')

rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
rgb[...,0] = np.nanmean(lay8[..., 4:8], 2)
rgb[...,1] = np.nanmean(lay8[..., 2:4], 2)
rgb[...,2] = np.nanmean(lay8[..., 0:2], 2)
# rgb[..., 0] = rgb[..., 0]**0.5
plt.imsave(pat+'mnn.png', rgb, origin='lower')
rgbw = whiten_image(rgb)
plt.imsave(pat+'mnnw.png', rgbw, origin='lower')

## more red

ii = 8
hdu0 = fits.open(pat+path[ii])
hdu0 = crval_fix1(hdu0)
img = hdu0[1].data
layers = np.zeros((img.shape[0], img.shape[1], len(path)))
layers[:, :, ii] = img
hdr0 = hdu0[1].header
hdu0.close()
for ii in range(len(path)):
    if ii == 8:
        hdu = fits.open(pat + path[ii])
    else:
        hdu = fits.open(pat+path[ii])
        hdu = crval_fix1(hdu)
    hdu[1].data = hole_func_fill(hdu[1].data)
    img, _ = reproject_interp(hdu[1], hdr0)
    layers[..., ii] = level_adjust(img, factor=2)

lay8 = np.zeros((img.shape[0], img.shape[1], 8))
lay8[..., 0] = np.nanmean(layers[..., 0:2], 2)
lay8[..., 1] = np.nanmean(layers[..., 2:4], 2)
lay8[..., 2] = np.nanmean(layers[..., 4:6], 2)
lay8[..., 3] = np.nanmean(layers[..., 6:8], 2)
for ii in range(4):
    lay8[..., ii+4] = layers[..., ii+8]

for ii in range(8):
    plt.subplot(2,4, ii + 1)
    plt.imshow(lay8[..., ii], cmap='gray')
    plt.axis('off')

rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
rgb[...,0] = np.nanmean(lay8[..., 4:8], 2)
rgb[...,1] = np.nanmean(lay8[..., 2:4], 2)
rgb[...,2] = np.nanmean(lay8[..., 0:2], 2)
# rgb[..., 0] = rgb[..., 0]**0.5
plt.imsave(pat+'mnn2.png', rgb, origin='lower')
rgbw = whiten_image(rgb)
plt.imsave(pat+'mnnw2.png', rgbw, origin='lower')


rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
rgb[...,0] = lay8[..., 3]
rgb[...,1] = np.nanmean(lay8[..., 1:3], 2)
rgb[...,2] = lay8[..., 0]
# rgb[..., 0] = rgb[..., 0]**0.5
plt.imsave(pat+'nircam2.png', rgb, origin='lower')
rgbw = whiten_image(rgb)
plt.imsave(pat+'nircam2w.png', rgbw, origin='lower')

flt = [200, 300, 335, 360]
col = matplotlib.cm.jet(flt / np.max(flt))[:, :3]  # [:, ::-1]
rgbf = assign_colors(rgb, col)
rgbf = blc_image(rgbf)
plt.imsave(pat+'nircam2filt.png', rgbf, origin='lower')


rgb[...,0] = lay8[..., 7]
rgb[...,1] = np.nanmean(lay8[..., 5:7], 2)
rgb[...,2] = lay8[..., 4]
# rgb[..., 0] = rgb[..., 0]**0.5
plt.imsave(pat+'miri2.png', rgb, origin='lower')
rgbw = whiten_image(rgb)
plt.imsave(pat+'miri2w.png', rgbw, origin='lower')

flt = [770, 1000, 1130, 2100]
col = matplotlib.cm.jet(flt / np.max(flt))[:, :3]  # [:, ::-1]
rgbf = assign_colors(rgb, col)
rgbf = blc_image(rgbf)

flt = [200, 300, 335, 360, 770, 1000, 1130, 2100]
col = matplotlib.cm.jet(flt / np.max(flt))[:, :3]
rgbf = assign_colors(lay8, col)
rgbf = blc_image(rgbf)
rgbf = whiten_image(rgbf)
# rgbf = reduce_color(rgbf, 2)
plt.imsave(pat+'filt.png', rgbf, origin='lower')
## core


ii = 8
hdu0 = fits.open(pat+path[ii])
hdu0 = crval_fix1(hdu0)
img = hdu0[1].data
layers = np.zeros((350, 300, len(path)))
# layers[:, :, ii] = img
hdr0 = hdu0[1].header
hdu0.close()
for ii in range(len(path)):
    if ii == 8:
        hdu = fits.open(pat + path[ii])
    else:
        hdu = fits.open(pat+path[ii])
        hdu = crval_fix1(hdu)
    hdu[1].data = hole_func_fill(hdu[1].data)
    img, _ = reproject_interp(hdu[1], hdr0)
    layers[..., ii] = level_adjust(img[800:1150, 875:1175], factor=2)

lay8 = np.zeros((350, 300, 8))
lay8[..., 0] = np.nanmean(layers[..., 0:2], 2)
lay8[..., 1] = np.nanmean(layers[..., 2:4], 2)
lay8[..., 2] = np.nanmean(layers[..., 4:6], 2)
lay8[..., 3] = np.nanmean(layers[..., 6:8], 2)
for ii in range(4):
    lay8[..., ii+4] = layers[..., ii+8]

for ii in range(8):
    plt.subplot(2,4, ii + 1)
    plt.imshow(lay8[..., ii], cmap='gray')
    plt.axis('off')

rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
rgb[...,0] = np.nanmean(lay8[..., 4:8], 2)
rgb[...,1] = np.nanmean(lay8[..., 2:4], 2)
rgb[...,2] = np.nanmean(lay8[..., 0:2], 2)
# rgb[..., 0] = rgb[..., 0]**0.5
plt.imsave(pat+'core2.png', rgb, origin='lower')
rgbw = whiten_image(rgb)
plt.imsave(pat+'corew2.png', rgbw, origin='lower')

rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
rgb[...,0] = lay8[..., 3]
rgb[...,1] = np.nanmean(lay8[..., 1:3], 2)
rgb[...,2] = lay8[..., 0]
# rgb[..., 0] = rgb[..., 0]**0.5
plt.imsave(pat+'core2nircam.png', rgb, origin='lower')
rgbw = whiten_image(rgb)
plt.imsave(pat+'corew2nircam.png', rgbw, origin='lower')

rgb[...,0] = lay8[..., 7]
rgb[...,1] = np.nanmean(lay8[..., 5:7], 2)
rgb[...,2] = lay8[..., 4]
# rgb[..., 0] = rgb[..., 0]**0.5
plt.imsave(pat+'core2miri.png', rgb, origin='lower')
rgbw = whiten_image(rgb)
plt.imsave(pat+'corew2miri.png', rgbw, origin='lower')
