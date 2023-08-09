from astro_utils import *
from astro_fill_holes import *
from bot_grabber import level_adjust
import os

#TODO: clean small holes fast without conv, remove red background

os.chdir('/media/innereye/My Passport/Data/JWST/data/NGC-3324/')
path = list_files('/media/innereye/My Passport/Data/JWST/data/NGC-3324/', '*_i2d.fits')
filt = filt_num(path)
order = np.argsort(filt)
path = np.asarray(path)[order]
##
# hdu = fits.open(path[6])
# img = level_adjust(hdu[1].data)
# plt.imshow(img, origin='lower')
# hdu0 = hdu = fits.open(path[0])
layers = reproject([path[6], path[1], path[0]])
for ii in [0, 1, 2]:
    layers[..., ii] = level_adjust(layers[..., ii])
plt.imshow(layers, origin='lower')
##
shifty = 21
shiftx = -4
fix = layers.copy()
fix[330+shifty:, shiftx+1606:shiftx+2130, 0] = layers[330:-shifty, 1606-shiftx:2130-shiftx, 0]
# plt.imshow(fix[300:800,1500:2200, :], origin='lower')
plt.imshow(fix, origin='lower')
plt.title(f'{shiftx} {shifty}')
##
template = 'jw02731-o002_t017_miri_f770w_i2d.fits'
# remove the template from the path
# path = path[:project_to] + path[project_to+1:]
hdu_temp = fits.open(template)
hdu_temp[1].header['CRVAL1'] = 159.22117
hdu_temp[1].header['CRVAL2'] = -58.61775
layers = np.ndarray((hdu_temp[1].shape[0], hdu_temp[1].shape[1], 3))
for ii, pp in enumerate(['./jw02731-o002_t017_miri_f770w_i2d.fits',
                        './jw02731-o001_t017_nircam_clear-f187n_i2d.fits',
                        './jw02731-o001_t017_nircam_clear-f090w_i2d.fits']):
    hdu = fits.open(pp)
    if ii == 0:
        layers[:, :, ii] = hdu[1].data
    else:
        reproj, _ = reproject_interp(hdu[1], hdu_temp[1].header)
        layers[:, :, ii] = reproj
for ii in [0, 1, 2]:
    layers[..., ii] = level_adjust(layers[..., ii])
plt.imshow(layers, origin='lower')
##

shifty = 20
shiftx = -4
fix = layers.copy()
fix[330+shifty:, shiftx+1606:shiftx+2130, 0] = layers[330:-shifty, 1606-shiftx:2130-shiftx, 0]
# plt.imshow(fix[300:800,1500:2200, :], origin='lower')
plt.imshow(fix, origin='lower')
plt.title(f'{shiftx} {shifty}')
##
project_to = 6
template = path[project_to]  # 'jw02731-o002_t017_miri_f770w_i2d.fits'
# remove the template from the path
# path = path[:project_to] + path[project_to+1:]
hdu_temp = fits.open(template)
hdu_temp[1].header['CRVAL1'] = 159.22117
hdu_temp[1].header['CRVAL2'] = -58.61775
layers = np.ndarray((hdu_temp[1].shape[0], hdu_temp[1].shape[1], len(path)))
for ii, pp in enumerate(path):
    hdu = fits.open(pp)
    if ii == project_to:
        layers[:, :, ii] = hdu[1].data
    else:
        reproj, _ = reproject_interp(hdu[1], hdu_temp[1].header)
        layers[:, :, ii] = reproj
##
factor = 2
for ii in range(layers.shape[2]):
    layers[..., ii] = level_adjust(layers[..., ii], factor=factor)
##
for lay in range(6):
    xy = hole_xy(layers[:, :, lay])
    size = hole_size(layers[:, :, lay], xy, plot=False)
    layers[:, :, lay] = hole_disk_fill(layers[:, :, lay], xy, size, larger_than=0, allowed=2/3)
    print(lay)
# plt.imshow(layers, origin='lower')
##
ib = [0, 1, 2]
ir = [6, 7, 8, 9]
ig = [3, 4 ,5]
iii = [ir, ig, ib]
rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
for ll in range(3):
    lay = np.mean(layers[..., iii[ll]], 2)
    lay = lay * 255
    rgb[:,:,ll] = lay
rgb = rgb.astype('uint8')

shifty = 20
shiftx = -4
rgb[0+shifty:, shiftx+1606:shiftx+2198, 0] = rgb[0:-shifty, 1606-shiftx:2198-shiftx, 0]

plt.figure()
plt.imshow(rgb, origin='lower')
plt.imsave(f'nircam+miri{factor}.png', rgb, origin='lower')

# plt.show()
# plt.imsave('right_finger_tot.png', np.flipud(np.fliplr(rgbt)), origin='lower')
##
#
# ncol = np.floor(layers.shape[-1] / 3)
# ib = np.arange(0,ncol).astype(int)
# ir = np.arange(layers.shape[-1]-ncol, layers.shape[-1]).astype(int)
# ig = np.arange(ib[-1]+1, ir[0]).astype(int)
# iii = [ir, ig, ib]
# rgbt = np.zeros((layers.shape[1], layers.shape[0], 3))
# for ll in range(3):
#     lay = np.mean(layers[:, :, iii[ll]], axis=2)
#     lay = lay * 255
#     rgbt[:,:,ll] = lay.T
#     print('done '+str(ll))
# rgbt = rgbt.astype('uint8')
# plt.figure()
# plt.imshow(rgbt)
# plt.show()