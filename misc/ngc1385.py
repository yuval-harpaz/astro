from astro_utils import *
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from reproject import reproject_interp
from astro_fill_holes import *
import pickle
import os

os.chdir('/home/innereye/JWST/ngc1385/')
path = list_files('/home/innereye/JWST/ngc1385/', '*miri*2d.fits')

filt = filt_num(path)
order = np.argsort(filt)
path = list(np.asarray(path)[order])

# crop = [3800,5000,5600,6800]
margins = 100

pkl = 'crop.pkl'
if os.path.isfile(pkl):
    layers = np.load(pkl, allow_pickle=True)
else:
    layers = np.zeros((1072, 729, len(path)))
    for ii in range(len(path)):
        print('start :' + str(ii))
        if ii == 0:
            hdu0 = fits.open(path[ii])
            # ref, ref_pos, ref_pix = crop_fits(hdu0[1], [6200, 4400], [1200, 1200])  # [4400, 6200]
            orig = hdu0[1].copy()
            ref, ref_pos, ref_pix = crop_fits(orig, [850, 650], [1000+margins, 700+margins])  # [4400, 6200]
            img = ref.data
            xy = hole_xy(img, x_stddev=6)
            size = hole_size(img, xy, plot=False)
            img = hole_disk_fill(img, xy, size, larger_than=2, allowed=0.5)
            # img = hole_conv_fill(img, n_pixels_around=6, ringsize=15, clean_below=1)
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
            # img = hole_conv_fill(img, n_pixels_around=3, ringsize=15, clean_below_local=0.75, clean_below=0.75)
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
    with open(pkl, 'wb') as f:
        pickle.dump(layers, f)


# layers = np.load('crop.pkl', allow_pickle=True)
# layers = optimize_xy(layers)[2]
# ng = 2
# nudged = layers.copy()
# nudged[ng:,ng:,1] = nudged[:-ng,:-ng,1]
# nudged[ng:,ng:,0] = nudged[:-ng,:-ng,0]
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

    # if ii == 0:
    #     img = img*2
    # elif ii == 2:
    #     img = img*0.8
    # else:
    #     img = img*1.3
    img = img / (med * meds) * 255
    img[img > 255] = 255
    img[img < 0] = 0
    # plt.subplot(2, 3, ii+1)
    # plt.imshow(img, cmap='gray', origin='lower')
    # plt.title(path[ii][-14:-9])
    # plt.axis('off')
    rgbt[..., 2-ii] = img

rgbt = rgbt.astype('uint8')
# rgbt = rgbt[50:-50, 50:-50,:]

plt.figure()
plt.imshow(np.flipud(np.fliplr(rgbt)), origin='lower')
plt.show()


plt.imsave('miri_rgb.png', np.flipud(np.fliplr(rgbt)), origin='lower')

##
# meds = 3
layers = np.load(pkl, allow_pickle=True)
meds = 1.5
total = np.zeros(layers.shape[:2])
c = 0
b = []
for ii in [0, 1, 2, 3]:  # range(layers.shape[2]):
    c += 1
    img = layers[:, :, ii].copy()
    img = img ** 0.5
    img[img == 0] = np.nan
    med = np.nanmedian(img)
    img[np.isnan(img)] = 0
    img = img - (med / meds)
    img = img / (med * meds) * 255
    img[img > 255] = 255
    img[img < 0] = 0
    # plt.subplot(2, 3, ii+1)
    # plt.imshow(img, cmap='gray', origin='lower')
    # plt.title(path[ii][-14:-9])
    # plt.axis('off')
    if b == []:
        b = img
    total += img
r = img


total = total / c
# layers = mosaic(path,method='layers')
rgbt = np.zeros((total.shape[0], total.shape[1], 3))
rgbt[..., 0] = r
rgbt[..., 1] = total  # *3-r-b
rgbt[..., 2] = b

rgbt = rgbt.astype('uint8')
# rgbt = rgbt[50:-50, 50:-50,:]
plt.figure()
plt.imshow(np.flipud(np.fliplr(rgbt)), origin='lower')
plt.show()
plt.imsave('miri_tot.png', np.flipud(np.fliplr(rgbt)), origin='lower')

tot = total.copy()
tot[tot < 45] = 45
tot = tot-45
tot = tot/(255-45)
plt.figure()
plt.imshow(tot, origin='lower', cmap='hot')
plt.show()
plt.imsave('miri_tot.png', tot, origin='lower', cmap='hot')


## NIRCam

os.chdir('/home/innereye/JWST/ngc1385/')
path = list_files('/home/innereye/JWST/ngc1385/', '*nircam*2d.fits')

# filt = filt_num(path)
# order = np.argsort(filt)
# path = list(np.asarray(path)[order])

# crop = [3800,5000,5600,6800]
margins = 100

pkl = 'cropcam.pkl'
if os.path.isfile(pkl):
    layers = np.load(pkl, allow_pickle=True)
else:
    layers = np.zeros((1777, 1180, len(path)))
    for ii in [3, 2, 1, 0]:
        print('start :' + str(ii))
        if ii == 3:
            hdu0 = fits.open(path[ii])
            # ref, ref_pos, ref_pix = crop_fits(hdu0[1], [6200, 4400], [1200, 1200])  # [4400, 6200]
            orig = hdu0[1].copy()
            ref, ref_pos, ref_pix = crop_fits(orig, [1000, 1400], [1920+margins, 1080+margins])  # [4400, 6200]
            img = ref.data
            # xy = hole_xy(img, x_stddev=6)
            # size = hole_size(img, xy, plot=False)
            # img = hole_disk_fill(img, xy, size, larger_than=2, allowed=0.5)
            # img = hole_conv_fill(img, n_pixels_around=6, ringsize=15, clean_below=1)
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
            # xy = hole_xy(img, x_stddev=6)
            # size = hole_size(img, xy, plot=False)
            # print('area = '+str(hdu[1].header['PIXAR_A2']))
            # print('prct 95 = '+str(np.round(np.percentile(np.nanmax(size, axis=1), 95), 1)))
            # img = hole_disk_fill(img, xy, size, larger_than=2, allowed=0.5)
            # img = hole_conv_fill(img, n_pixels_around=3, ringsize=15, clean_below_local=0.75, clean_below=0.75)
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
    with open(pkl, 'wb') as f:
        pickle.dump(layers, f)


# layers = np.load('crop.pkl', allow_pickle=True)
# layers = optimize_xy(layers)[2]
# ng = 2
# nudged = layers.copy()
# nudged[ng:,ng:,1] = nudged[:-ng,:-ng,1]
# nudged[ng:,ng:,0] = nudged[:-ng,:-ng,0]
meds = 3
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
    # plt.subplot(2, 3, ii+1)
    # plt.imshow(img, cmap='gray', origin='lower')
    # plt.title(path[ii][-14:-9])
    # plt.axis('off')
    rgbt[..., 2-ii] = img

rgbt = rgbt.astype('uint8')
# rgbt = rgbt[50:-50, 50:-50,:]

plt.figure()
plt.imshow(np.flipud(np.fliplr(rgbt)), origin='lower')
plt.show()


plt.imsave('nircam_rgb.png', np.flipud(np.fliplr(rgbt)), origin='lower')

##
meds = 4
layers = np.load(pkl, allow_pickle=True)
total = np.zeros(layers.shape[:2])
c = 0
b = []
for ii in [0, 1, 2, 3]:  # range(layers.shape[2]):
    c += 1
    img = layers[:, :, ii].copy()
    img = img ** 0.5
    img[img == 0] = np.nan
    med = np.nanmedian(img)
    img[np.isnan(img)] = 0
    img = img - (med / meds)
    img = img / (med * meds) * 255
    img[img > 255] = 255
    img[img < 0] = 0
    # plt.subplot(2, 3, ii+1)
    # plt.imshow(img, cmap='gray', origin='lower')
    # plt.title(path[ii][-14:-9])
    # plt.axis('off')
    if b == []:
        b = img
    total += img
r = img


total = total / c
# layers = mosaic(path,method='layers')
rgbt = np.zeros((total.shape[0], total.shape[1], 3))
rgbt[..., 0] = r
rgbt[..., 1] = total  # *3-r-b
rgbt[..., 2] = b

rgbt = rgbt.astype('uint8')
# rgbt = rgbt[50:-50, 50:-50,:]
plt.figure()
plt.imshow(np.flipud(np.fliplr(rgbt)), origin='lower')
plt.show()
plt.imsave('nircam_tot.png', np.flipud(np.fliplr(rgbt)), origin='lower')

tot = total.copy()
# tot[tot < 45] = 45
# tot = tot-45
# tot = tot/(255-45)
plt.figure()
plt.imshow(tot, origin='lower', cmap='hot')
plt.show()
plt.imsave('nircam_tot.png', tot, origin='lower', cmap='hot')