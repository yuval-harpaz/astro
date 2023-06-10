from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from reproject import reproject_interp
from astro_fill_holes import *
import pickle
import os

os.chdir('/home/innereye/JWST/ngc2070')
path = ['jw02729-o001_t001_nircam_clear-f090w_i2d.fits',
        'jw02729-o001_t001_nircam_clear-f200w_i2d.fits',
        'jw02729-o001_t001_nircam_clear-f444w_i2d.fits']
# filt = filt_num(path)
# crop = [3800,5000,5600,6800]
margins = 100
fac = 3
layers = np.zeros((1080 * fac + margins, 1920 * fac + margins, len(path)))
for ii in range(len(path)):
    print('start :' + str(ii))
    if ii == 0:
        hdu0 = fits.open(path[ii])
        # ref, ref_pos, ref_pix = crop_fits(hdu0[1], [6200, 4400], [1200, 1200])  # [4400, 6200]
        orig = hdu0[1].copy()
        ref, ref_pos, ref_pix = crop_fits(orig, [7500, 4500], [fac*1080+margins, fac*1920+margins])  # [4400, 6200]
        img = ref.data
        xy = hole_xy(img, x_stddev=6)
        size = hole_size(img, xy, plot=False)
        img = hole_disk_fill(img, xy, size, larger_than=2, allowed=0.5)
        img = hole_conv_fill(img, n_pixels_around=0, ringsize=15, clean_below=1)
        # img = img[crop[0]:crop[1],crop[2]:crop[3]]
        # img = fill_holes(img, pad=1, hole_size=50)
        hdr0 = ref.header
        del hdu0
    else:
        hdu = fits.open(path[ii])

        wcs = WCS(hdu[1].header)
        pix = wcs.wcs_world2pix(ref_pos, 0)
        pix = np.round(np.asarray(pix))
        size = 2 * (pix[1, :] - pix[0, :])
        hdu[1], _, _ = crop_fits(hdu[1], pix[1, :], size)
        img = hdu[1].data
        xy = hole_xy(img, x_stddev=6)
        size = hole_size(img, xy, plot=False)
        print('area = '+str(hdu[1].header['PIXAR_A2']))
        print('prct 95 = '+str(np.round(np.percentile(np.nanmax(size, axis=1), 95), 1)))
        img = hole_disk_fill(img, xy, size, larger_than=2, allowed=0.5)
        img = hole_conv_fill(img, n_pixels_around=3, ringsize=15, clean_below_local=0.75, clean_below=0.75)
        # plt.figure();plt.imshow(img, origin='lower');plt.clim(0,1000);plt.show(block=False)
        hdu[1].data = img
        img, _ = reproject_interp(hdu[1], hdr0)
        # img = img[crop[0]:crop[1],crop[2]:crop[3]]

    if img.shape[0] == 0:
        raise Exception('bad zero')
    # img[img == 0] = np.nan
    # med = np.nanmedian(img)
    # img[np.isnan(img)] = 0
    layers[:,:,ii] = img

# layers = np.load('layers.pkl', allow_pickle=True)
# layers = optimize_xy(layers, tests=200, square_size=1500)[2]
peaks = []
for ii in [0,1,2]:
    peaks.append(find_peaks2d(layers[1000:3000, 1000:3000, ii]))

minxy = np.zeros((len(peaks[0]), 2))
for kk in range(len(peaks[0])):
    mindist = np.argmin((peaks[2][:, 0]-peaks[0][kk, 0])**2+(peaks[2][:, 1]-peaks[0][kk, 1])**2)
    minxy[kk,:] = peaks[2][mindist,:] - peaks[0][kk,:]

meds = 6
total = np.zeros(layers.shape[:2])
c = 0
b = []
for ii in range(layers.shape[2]):
    c += 1
    img = layers[:, :, ii]
    img[img == 0] = np.nan
    med = np.nanmedian(img)
    img[np.isnan(img)] = 0
    img = img - (med / meds)
    img = img / (med * meds) * 255
    img[img > 255] = 255
    img[img < 0] = 0
    if b == []:
        b = img
    total += img
r = img
total = total / c
rgbt = np.zeros((total.shape[0], total.shape[1], 3))
rgbt[..., 0] = r
rgbt[..., 1] = total
rgbt[..., 2] = b
rgbt = rgbt.astype('uint8')
rgbt = rgbt[50:-50, 50:-50,:]

plt.figure()
plt.imshow(rgbt, origin='lower')
plt.show()


# plt.imsave('carina_bay_5.png',rgbt,origin='lower')
