from astro_utils import *
from astro_fill_holes import *
from bot_grabber import level_adjust
import os

#TODO: clean small holes fast without conv, remove red background

os.chdir('/home/innereye/JWST/ngc1433')
path = list_files('/home/innereye/JWST/ngc1512/', '*_i2d.fits')
filt = filt_num(path)
order = np.argsort(filt)
path = np.asarray(path)[order]
# miri
# miri = path[4:]
margins = 100
# center = [770, 1000]
# size = [int(1920*1.5), int(1080*1.5)] # a narrow image

# plt.figure()
for ii in range(len(path)):
    if ii == 0:

        hdu0 = fits.open(path[ii])
        img = hdu0[1].data
        layers = np.zeros((img.shape[0], img.shape[1], len(path)))
        # coord = [2150, 4000]
        # ref, ref_pos, ref_pix = crop_fits(orig, coord, [], [], crop=False)
        # img = hole_conv_fill(img, n_pixels_around=None, ringsize=10, clean_below=0)
        img = level_adjust(img)
        hdr0 = hdu0[1].header
        hdu0.close()
    else:
        hdu = fits.open(path[ii])
        # wcs = WCS(hdu[1].header)
        # pix = wcs.wcs_world2pix(ref_pos, 0)
        # pix = np.round(np.asarray(pix))
        # sz = 2 * (pix[1, :] - pix[0, :])
        # hdu[1], _, _ = crop_fits(hdu[1], pix[1, :], sz)
        img = hdu[1].data
        # img = hole_conv_fill(img, n_pixels_around=None, ringsize=10, clean_below=0)
        img = level_adjust(img)
        hdu[1].data = img
        img, _ = reproject_interp(hdu[1], hdr0)
    # plt.subplot(3, 3, ii+1)
    # plt.imshow(img)
    layers[:, :, ii] = img
# plt.show(block=False)

total = np.zeros(layers.shape[:2])
c = 0
b = []
for ii in range(layers.shape[-1]):  # range(layers.shape[2]):
    c += 1
    img = layers[:, :, ii]
    if b == []:
        b = img
    total += img
r = img
total = total / c

rgbt = np.zeros((total.shape[1], total.shape[0], 3))
rgbt[..., 0] = r.T * 255
rgbt[..., 1] = total.T * 255  # *3-r-b
rgbt[..., 2] = b.T * 255

rgbt = rgbt.astype('uint8')
# rgbt = rgbt[margins:-margins, margins:-margins,:]
plt.figure()
plt.imshow(rgbt)
plt.show()


ncol = np.floor(layers.shape[-1] / 3)
ib = np.arange(0,ncol).astype(int)
ir = np.arange(layers.shape[-1]-ncol, layers.shape[-1]).astype(int)
ig = np.arange(ib[-1]+1, ir[0]).astype(int)
iii = [ir, ig, ib]
rgbt = np.zeros((layers.shape[1], layers.shape[0], 3))
for ll in range(3):
    lay = np.zeros((layers.shape[0], layers.shape[1]))
    for mm in range(layers.shape[0]):
        for nn in range(layers.shape[1]):
            lay[mm, nn] = np.nanmean(layers[mm,nn,iii[ll]])
    lay = lay * 255
    rgbt[:,:,ll] = lay.T
rgbt = rgbt.astype('uint8')
plt.figure()
plt.imshow(rgbt)
plt.show()
# plt.imsave('right_finger_tot.png', np.flipud(np.fliplr(rgbt)), origin='lower')


ncol = np.floor(layers.shape[-1] / 3)
ib = np.arange(0,ncol).astype(int)
ir = np.arange(layers.shape[-1]-ncol, layers.shape[-1]).astype(int)
ig = np.arange(ib[-1]+1, ir[0]).astype(int)
iii = [ir, ig, ib]
rgbt = np.zeros((layers.shape[1], layers.shape[0], 3))
for ll in range(3):
    lay = np.mean(layers[:, :, iii[ll]], axis=2)
    lay = lay * 255
    rgbt[:,:,ll] = lay.T
    print('done '+str(ll))
rgbt = rgbt.astype('uint8')
plt.figure()
plt.imshow(rgbt)
plt.show()