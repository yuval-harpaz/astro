# %matplotlib qt
from astro_utils import *
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from reproject import reproject_interp
from astro_fill_holes import *
import pickle
import os

kernel = Gaussian2DKernel(x_stddev=3)
os.chdir('/home/innereye/JWST/SGAS/')
path = list_files('/home/innereye/JWST/SGAS/', '*nircam*2d.fits')

filt = filt_num(path)
order = np.argsort(filt)
path = list(np.asarray(path)[order])
# crop = 2900:4100,2500:3700
margins = 100

pkl = 'crop.pkl'
if os.path.isfile(pkl):
    layers = np.load(pkl, allow_pickle=True)
else:
    layers = np.zeros((1300, 1300, len(path)))
    for ii in range(len(path)):
        print('start :' + str(ii))
        if ii == 0:
            hdu0 = fits.open(path[ii])
            orig = hdu0[1].copy()
            ref, ref_pos, ref_pix = crop_fits(orig, [3100, 3500], [1200+margins, 1200+margins])  # [4400, 6200]
            img = ref.data
            hdr0 = ref.header
            del hdu0
        else:
            hdu = fits.open(path[ii])

            wcs = WCS(hdu[1].header)
            pix = wcs.wcs_world2pix(ref_pos, 0)
            pix = np.round(np.asarray(pix))
            size = 2 * (pix[1, :] - pix[0, :])
            hdu[1], _, _ = crop_fits(hdu[1], pix[1, :], size)
            img, _ = reproject_interp(hdu[1], hdr0)
        if img.shape[0] == 0:
            raise Exception('bad zero')
        layers[:,:,ii] = img
    with open(pkl, 'wb') as f:
        pickle.dump(layers, f)
##
meds = 1.5
# total = np.zeros(layers.shape[:2])
rgbt = np.zeros((layers.shape[0], layers.shape[1], 3))
c = 0
b = []
for ii in [0, 1, 2]:  # range(layers.shape[2]):
    c += 1
    img = layers[:, :, ii]
    img = img ** 0.5
    img[img == 0] = np.nan
    med = np.nanmedian(img)
    img[np.isnan(img)] = 0
    img = img - (med / meds)
    img = img / (med * meds) * 255
    img[img > 255] = 255
    img[img < 0] = 0
    rgbt[..., 2-ii] = img

rgbt = rgbt.astype('uint8')
# rgbt = rgbt[50:-50, 50:-50,:]

plt.figure()
plt.imshow(np.flipud(np.fliplr(rgbt)), origin='lower')
plt.show()


# plt.imsave('miri_rgb.png', np.flipud(np.fliplr(rgbt)), origin='lower')

##
# meds = 3
layers = np.load(pkl, allow_pickle=True)
meds = 1.5
total = np.zeros(layers.shape[:2])
layern = layers.copy()
c = 0
b = []
prc = 5
x1 = 650
x2 = 1100
extra = 0.01
for ii in [0,1,2,3,4]:  # range(layers.shape[2]):
    c += 1
    img = layers[:, :, ii].copy()
    md = np.nanpercentile(img, prc)
    for y in range(img.shape[0]):
        mdy = np.nanpercentile(img[y, x1:x2], prc)
        mdy = np.nanmean(img[y, x1:x2] < mdy)
        img[y, :] = img[y, :] - mdy + md
    img = img ** 0.5
    img[img == 0] = np.nan
    med = np.nanmedian(img)
    img[np.isnan(img)] = 0
    img = img - (med / meds)*1.5 - extra
    img = img / (med * meds) * 255
    img[img > 255] = 255
    img[img < 0] = 0
    if b == []:
        b = img
    layern[:, :, ii] = img
r = img
isr = True
for rb in [r, b]:
    rb = convolve(rb, kernel=kernel)
    rb[rb > 255] = 255
    rb[rb < 0] = 0
    if isr:
        r = rb
        isr = False
    else:
        b = rb
total = np.nanmean(layern, axis=2)

# total = total / c
# layers = mosaic(path,method='layers')
rgbt = np.zeros((total.shape[0], total.shape[1], 3))
rgbt[..., 0] = r
rgbt[..., 1] = total  # *3-r-b
rgbt[..., 2] = b

rgbt = rgbt.astype('uint8')
# rgbt = rgbt[50:-50, 50:-50,:]
plt.figure()
h = plt.imshow(rgbt, origin='lower')  # , vmin=200, vmax=255
plt.show()
# h.set_clim(vmin=200, vmax=255)
# plt.xlim(650, 900)
# plt.ylim(650, 900)

# plt.imsave('miri_tot.png', np.flipud(np.fliplr(rgbt)), origin='lower')
##
# tot = total.copy()
# tot[tot < 45] = 45
# tot = tot-45
# tot = tot/(255-45)
# plt.figure()
# plt.imshow(tot, origin='lower', cmap='hot')
# plt.show()
# plt.imsave('miri_tot.png', tot, origin='lower', cmap='hot')



