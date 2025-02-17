from astro_utils import *
os.chdir('/media/innereye/KINGSTON/JWST/data/')
files = glob('NGC4485*/*.fits')
for f in files:
    hdu = fits.open(f)
    print(hdu[1].data.shape)
mos = mosaic(files, method='layers')  # , xy=[], size=[], clip=[], method='overwrite', plot=False, log=None, fill=True, subtract=None):

# rgb[..., 0] = level_adjust(np.nanmean(mos[..., [0, 2]], 2), factor=2)
# rgb[..., 2] = level_adjust(np.nanmean(mos[..., [1, 3]], 2), factor=2)
prct = [[], []]
prct[0] = np.nanpercentile(list(mos[..., 0].flatten()) +
                           list(mos[..., 2].flatten()), [1, 95])
prct[1] = np.nanpercentile(list(mos[..., 1].flatten()) +
                           list(mos[..., 3].flatten()), [1, 95])
plt.figure()
for ii in range(4):
    plt.subplot(2,2,ii+1)
    img = mos[..., ii] / prct[ii%2][1]
    img[img > 1] = 1
    img[img < 0] = 0
    plt.imshow(img, origin='lower')
    plt.title(files[ii])

rgb = np.zeros((mos.shape[0], mos.shape[1], 3))
rgb[..., 0] = np.nanmax(mos[..., [0, 2]], 2)
rgb[..., 0] = rgb[..., 0] - prct[0][0]
rgb[..., 0] = rgb[..., 0] / (prct[0][1] - prct[0][0])
rgb[..., 2] = np.nanmax(mos[..., [1, 3]], 2)
rgb[..., 2] = rgb[..., 2] - prct[1][0]
rgb[..., 2] = rgb[..., 2] / (prct[1][1] - prct[1][0])
rgb[..., 1] = (rgb[..., 0] + rgb[..., 2]) / 2
rgb[rgb > 1] = 1
rgb[rgb < 0] = 0
rgb[np.isnan(rgb)] = 0
plt.imsave('/home/innereye/Pictures/NGC-4495-4490-prct.jpg', rgb, pil_kwargs={'quality': 95}, origin='lower')
# This failed because the two readings are in different orientation, I think. must interpolate

##
img = plt.imread('/media/innereye/KINGSTON/JWST/data/NGC4485-NGC4490-MIRI/manual.png')[..., :3]
gray = np.mean(img, 2)
black = gray == 1
for lay in [0, 2]:
    layer = img[..., lay]
    layer[black] = 0
    img[..., lay] = layer
img[..., 1] = (img[..., 0] + img[..., 2]) / 2
plt.imsave('/home/innereye/Pictures/NGC-4495-4490-blue.jpg', img, pil_kwargs={'quality': 95})
imgb = img.copy()
blue = img[..., 2]
for lay in [0, 1]:
    layer = img[..., lay]
    layer[layer < blue] = blue[layer < blue]
    img[..., lay] = layer
imgm = (img + imgb)/2
plt.imshow(imgm)
plt.imsave('/home/innereye/Pictures/NGC-4495-4490-ok.jpg', imgm, pil_kwargs={'quality': 95})
