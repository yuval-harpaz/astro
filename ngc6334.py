import matplotlib.pyplot as plt

from astro_utils import *
os.chdir('/media/innereye/KINGSTON/JWST/NGC6334')
files = glob('*N-*/*.fits')

mos = mosaic(files, method='layers', fill=True)  # , xy=[], size=[], clip=[], method='overwrite', plot=False, log=None, fill=True, subtract=None):
mos[mos == 0] = np.nan
flat = np.nanmin(mos, 2)
img = log(flat)
# mos[mos == 0] = np.nan
# img = level_adjust(log(mos), factor=1)
img = img-np.nanmin(img)
img[np.isnan(img)] = 0
img = img/np.max(img)
# img[img < 0] = 0
plt.imsave('tmp.jpg', img, origin='lower',  pil_kwargs={'quality': 95}, cmap='gray')
# plt.imsave('Mgray1log.jpg', img, origin='lower',  pil_kwargs={'quality': 95}, cmap='gray')

# os.chdir('/media/innereye/My Passport/Data/JWST/data/NGC0300MIRI/')
# path = glob('*fits')
# layers = reproject(path)
# for ii in [0, 1]:
#     layers[..., ii] = level_adjust(layers[..., ii])


# plt.figure()
# plt.imshow(layers[..., [1, 0, 0]])

# rgb = layers[..., [1, 0, 0]].copy()
# for row in range(rgb.shape[0]):
#     vec = rgb[row, :, 0].copy()
#     vec = vec[vec > 0]
#     rgb[row, :, 0] = rgb[row, :, 0] - np.median(vec)
#     print(row)

# rgb[..., 0] = rgb[..., 0] + np.min(rgb[..., 0])
# rgb[..., 0] = level_adjust(rgb[..., 0])
# rgb[..., 1] = (rgb[..., 0] + rgb[..., 2]) / 2
# plt.figure()
# plt.imshow(rgb)

# plt.imsave('ngc300.png', rgb, origin='lower')
