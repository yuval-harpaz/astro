from astro_utils import *
os.chdir(drive)
os.chdir('w51')
miri = glob('*miri*reprj_f2100*')
exp = ['jw06151-o002_t001_miri_f560w_i2d_reprj_f2100_nanfilled.fits', 
    'jw06151-o002_t001_miri_f1000w_i2d_reprj_f2100_nanfilled.fits',
    'jw06151-o002_t001_miri_f2100w_i2d_reprj_f2100_nanfilled.fits']
img = np.zeros((2060, 1722, 3))
for ii in range(3):
    hdu = fits.open(exp[ii])
    img[:, :, 2-ii] = hdu[0].data
img[np.isnan(img)] = 0
img[img < 0] = 0
img = img/img.max()
plt.imsave('miri.jpg', img, origin='lower', pil_kwargs={'compression': 'jpeg', 'quality': 95})

img = np.zeros((2060, 1722, 3))
for ii in range(3):
    hdu = fits.open(exp[ii])
    img[:, :, 2-ii] = level_adjust(hdu[0].data, factor=2)
# img[np.isnan(img)] = 0
# img[img < 0] = 0
# img = img/img.max()
plt.imsave('miri2.jpg', img, origin='lower', pil_kwargs={'compression': 'jpeg', 'quality': 95})

exp = ['jw06151-o001_t001_nircam_clear-f140m-merged_i2d_reprj_f140_nanfilled.fits', 'jw06151-o001_t001_nircam_clear-f360m-merged_i2d_reprj_f140_nanfilled.fits', 'jw06151-o001_t001_nircam_clear-f480m-merged_i2d_reprj_f140_nanfilled.fits']
img = np.zeros((5192, 11452, 3))
for ii in range(3):
    hdu = fits.open(exp[ii])
    img[:, :, 2-ii] = level_adjust(hdu[0].data, factor=2)
# img[np.isnan(img)] = 0
# img[img < 0] = 0
# img = img/img.max()
plt.imsave('clear2.jpg', img, origin='lower', pil_kwargs={'compression': 'jpeg', 'quality': 95})


exp = ['jw06151-o001_t001_nircam_clear-f140m-merged_i2d_reprj_f140_nanfilled.fits', 'jw06151-o001_t001_nircam_clear-f480m-merged_i2d_reprj_f140_nanfilled.fits', 'jw06151-o002_t001_miri_f1000w_i2d_reprj_f140_nanfilled.fits']
img = np.zeros((5192, 11452, 3))
for ii in range(3):
    hdu = fits.open(exp[ii])
    img[:, :, 2-ii] = level_adjust(hdu[0].data, factor=2)
# img[np.isnan(img)] = 0
# img[img < 0] = 0
# img = img/img.max()
plt.imsave('comb2.jpg', img, origin='lower', pil_kwargs={'compression': 'jpeg', 'quality': 95})


##
exp = ['jw06151-o001_t001_nircam_clear-f140m-merged_i2d_reprj_f140_nanfilled.fits',
       'jw06151-o001_t001_nircam_clear-f360m-merged_i2d_reprj_f140_nanfilled.fits',
       'jw06151-o001_t001_nircam_clear-f480m-merged_i2d_reprj_f140_nanfilled.fits',
       'jw06151-o002_t001_miri_f560w_i2d_reprj_f140_nanfilled.fits',
       'jw06151-o002_t001_miri_f770w_i2d_reprj_f140_nanfilled.fits',
       'jw06151-o002_t001_miri_f1000w_i2d_reprj_f140_nanfilled.fits']

filt = [1000, 770, 560, 480, 360, 140]
layers = np.zeros((5192, 11452, 6))
for ii in range(6):
    hdu = fits.open(exp[ii])
    data = hole_func_fill(hdu[0].data, func='max')
    layers[:, :, 5-ii] = level_adjust(data, factor=2)

rgb = assign_colors_by_filt(layers, np.array(filt))
rgb = rgb * 255
rgb = rgb.astype('uint8')
plt.imsave('miri+nircam.jpg', rgb, origin='lower', pil_kwargs={'compression': 'jpeg', 'quality': 95})

exp = ['jw06151-o001_t001_nircam_clear-f140m-merged_i2d_reprj_f140_nanfilled.fits', 'jw06151-o001_t001_nircam_clear-f360m-merged_i2d_reprj_f140_nanfilled.fits', 'jw06151-o001_t001_nircam_clear-f480m-merged_i2d_reprj_f140_nanfilled.fits']
filt = [480, 360, 140]
layers = np.zeros((5192, 11452, 3))
for ii in range(3):
    hdu = fits.open(exp[ii])
    data = hole_func_fill(hdu[0].data, func='max')
    layers[:, :, 2-ii] = level_adjust(data, factor=2)
rgb = assign_colors_by_filt(layers, np.array(filt))
rgb = rgb * 255
rgb = rgb.astype('uint8')
plt.imsave('nircam_filt2.jpg', rgb, origin='lower', pil_kwargs={'compression': 'jpeg', 'quality': 95})

# plt.imshow(layers[3250:, 1000:3750,:])
##
cropped = np.zeros((1942, 2750, 3))
for ii in range(3):
    hdu = fits.open(exp[ii])
    data = hole_func_fill(hdu[0].data[3250:, 1000:3750], func='max')
    data = deband_layer(data, win=101, prct=10, func=np.nanpercentile, flip=False, verbose=False)
    cropped[:, :, 2-ii] = level_adjust(data, factor=2)
rgb = assign_colors_by_filt(cropped, np.array(filt))
rgb = rgb * 255
rgb = rgb.astype('uint8')
plt.imsave('nircam_cropped_101.jpg', rgb, origin='lower', pil_kwargs={'compression': 'jpeg', 'quality': 95})

