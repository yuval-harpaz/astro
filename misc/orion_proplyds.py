from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from reproject import reproject_interp
from astro_fill_holes import *
import pickle


path = list_files('/home/innereye/JWST/Ori/', search='*.fits')
''' path includes these images:
['jw01288-o001_t011_nircam_clear-f140m/jw01288-o001_t011_nircam_clear-f140m_i2d.fits',
 'jw01288-o001_t011_nircam_clear-f182m/jw01288-o001_t011_nircam_clear-f182m_i2d.fits',
 'jw01288-o001_t011_nircam_clear-f210m/jw01288-o001_t011_nircam_clear-f210m_i2d.fits',
 'jw01288-o001_t011_nircam_clear-f300m/jw01288-o001_t011_nircam_clear-f300m_i2d.fits',
 'jw01288-o001_t011_nircam_clear-f335m/jw01288-o001_t011_nircam_clear-f335m_i2d.fits',
 'jw01288-o001_t011_nircam_clear-f480m/jw01288-o001_t011_nircam_clear-f480m_i2d.fits']
'''
mfilt = np.where(['m_i2d.fits' in x for x in path])[0][:-1]
path = [path[ii] for ii in mfilt]
meds = 4
# crop = [3800,5000,5600,6800]
margins = 100
layers = np.zeros((1080+margins, 1920+margins, 6))

if os.path.isfile('cropped.pkl'):
    layers = np.load('cropped.pkl', allow_pickle=True)
else:
    for ii in range(len(path)):
        print('start :' + str(ii))
        if ii == 0:
            hdu0 = fits.open(path[ii])
            # ref, ref_pos, ref_pix = crop_fits(hdu0[1], [6200, 4400], [1200, 1200])  # [4400, 6200]
            orig = hdu0[1].copy()
            ref, ref_pos, ref_pix = crop_fits(orig, [6760, 4140], [1080+margins, 1920+margins])  # [4400, 6200]
            img = ref.data
            xy = hole_xy(img, x_stddev=6)
            size = hole_size(img, xy, plot=False)
            img = hole_disk_fill(img, xy, size, larger_than=2, allowed=0.5)
            img = hole_conv_fill(img, n_pixels_around=6, ringsize=15, clean_below=1)
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
    with open('cropped.pkl', 'wb') as f:
        pickle.dump(layers, f)

total = np.zeros(layers.shape[:2])
plt.figure()
for ii in range(5):  # layers.shape[2]):
    img = layers[:, :, ii]
    img[img == 0] = np.nan
    med = np.nanmedian(img)
    img[np.isnan(img)] = 0
    img = img - (med / meds)
    img = img / (med * meds) * 255
    img[img > 255] = 255
    img[img < 0] = 0
    plt.subplot(2, 3, ii+1)
    plt.imshow(img, cmap='gray', origin='lower')
    if ii == 0:
        b = img
    elif ii == 4:
        r = img
    total += img

plt.show(block=False)


total = total / (ii+1)
# layers = mosaic(path,method='layers')
rgbt = np.zeros((total.shape[0], total.shape[1], 3))
rgbt[..., 0] = r
rgbt[..., 1] = total
rgbt[..., 2] = b

rgbt = rgbt.astype('uint8')
rgbt = rgbt[50:-50, 50:-50,:]
plt.figure()
plt.imshow(rgbt, origin='lower')
plt.show(block=False)


# plt.imsave('proplyd05.png',rgbt,origin='lower')
