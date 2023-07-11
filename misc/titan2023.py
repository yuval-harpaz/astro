from astro_utils import *
from astro_fill_holes import *
import os
files = ['jw01251-o003_t001_nircam_clear-f187n-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_f150w2-f164n-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f140m-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f444w-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f300m-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f182m-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f090w-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f480m-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f356w-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f150w-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f210m-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f460m-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f335m-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f250m-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f430m-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f070w-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f115w-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f200w-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_f444w-f466n-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_f150w2-f162m-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_f322w2-f323n-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f277w-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f212n-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f360m-sub160p_i2d.fits', 'jw01251-o003_t001_nircam_clear-f410m-sub160p_i2d.fits']
link = 'https://mast.stsci.edu/portal/Download/file/JWST/product/'

for fn in files:
    os.system('wget '+link+fn)

auto_plot('titan', exp='*.fits', png=True, pow=[1, 1, 1], pkl=True, resize=False, method='rrgggbb', plot=True)
filt = filt_num(files)
order = np.argsort(-filt)
filt = filt[order]
files = np.asarray(files)[order]
##
plt.figure()
for ii in range(len(files)):
    plt.subplot(5,5,ii+1)
    hdu = fits.open(files[ii])
    plt.imshow(level_adjust(hdu[1].data))
    plt.title(filt[ii])
    plt.axis('off')

##
files = np.asarray(files)[filt < 250]
##
layers = np.zeros((182,182,12))
plt.figure()
for ii in range(len(files)):
    plt.subplot(3,4,ii+1)
    hdu = fits.open(files[ii])
    plt.imshow(level_adjust(hdu[1].data))
    plt.title(filt[ii])
    plt.axis('off')
    print(hdu[1].data.shape)
    layers[..., ii] = hdu[1].data[:182, :182]
    # print(hdu[min(1, len(hdu)-1)].data)
##
rgb = np.zeros((182, 182, 3))
for lay in range(3):
    idx0 = lay*4
    idx1 = lay*4+4
    # print(f'{idx0} {idx1}')
    layer = layers[..., idx0:idx1].copy()
    layer = np.mean(layer, 2)
    layer = layer/layer.max()
    rgb[..., lay] = layer
rgb = rgb*255
rgb = rgb.astype('uint8')
plt.figure()
plt.imshow(rgb)
plt.axis('off')
plt.imsave('titan.png', rgb, origin='lower')
##

rgb = np.zeros((182, 182, 3))
for lay in range(3):
    idx0 = lay*4
    idx1 = lay*4+4
    # print(f'{idx0} {idx1}')
    layer = layers[..., idx0:idx1].copy()
    for ii in range(4):
        layer[..., ii] = level_adjust(layer[..., ii])
    layer = np.mean(layer, 2)
    # layer = layer/layer.max()
    rgb[..., lay] = layer
rgb = rgb*255
rgb = rgb.astype('uint8')
plt.figure()
plt.imshow(rgb)
plt.axis('off')
plt.imsave('titan_adjust.png', rgb, origin='lower')

##

plt.figure()
for ii in range(len(files)):
    plt.subplot(3,4,ii+1)
    hdu = fits.open(files[ii])
    plt.imshow(level_adjust(layers[45:135,45:135, ii]), origin='lower', cmap='gray')
    plt.title(filt[ii])
    plt.axis('off')
    plt.title(files[ii][25:-9])

##
rgb3 = np.zeros((90, 90, 3))
ch = [1, 7, 9]
i = 0
for c in ch:
    rgb3[..., i] = level_adjust(layers[45:135, 45:135, c])
    i += 1
rgb3 = rgb3*255
rgb3 = rgb3.astype('uint8')
plt.figure()
plt.imshow(rgb3)
plt.axis('off')
