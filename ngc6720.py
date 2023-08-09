import numpy as np

from astro_utils import *
_ = auto_plot('ring', exp='*.fits', png='factor1.png', pow=[2, 1, 1], pkl=False, method='rrgggbb', resize=False, plot=True, adj_args={'ignore0': True, 'factor': 1}, max_color=True)
##
path = list_files('/media/innereye/My Passport/Data/JWST/data/ring/')
layers = mosaic(path, method='layers', plot=False)
##
lim = [[0, 1],[-0.05, 0],[0,1], [0, 1],[-0.05, 0],[0,1]]
plt.figure()
for ii in range(6):
    plt.subplot(2,3,ii+1)
    plt.imshow(layers[..., ii])
    plt.title(f'{np.round(np.percentile(layers[..., ii], 10),2)} {np.round(np.percentile(layers[..., ii], 90))}')
    plt.clim(lim[ii])
##
ladj = layers.copy()
for ii in range(6):
    ladj[..., ii] = level_adjust(layers[...,ii].copy(), ignore0=True, factor=2)
ladj[ladj == 0] = np.nan
rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
# layers[layers <= 0] = np.nan
rgb[..., 0] = np.nanmean(ladj[..., [1, 4]], axis=2)**2
rgb[..., 1] = np.nanmean(ladj[..., [0, 3]], axis=2)
rgb[..., 2] = np.nanmean(ladj[..., [2, 5]], axis=2)
#
# for ii in range(3):
#     rgb[..., ii] = level_adjust(rgb[..., ii])
plt.figure()
plt.imshow(rgb)
##
plt.imsave('mosaic.png', rgb, origin='lower')


##
os.chdir('/media/innereye/My Passport/Data/JWST/ring_workshop/')
obj = Observations.query_object('NGC6720')
obj = obj[obj['instrument_name'] == 'NIRCAM/IMAGE']
for obs in obj:
    all = Observations.get_product_list(obs)
    filt = Observations.filter_products(all, extension='_i2d.fits')
    Observations.download_products(filt)


path = list_files('/media/innereye/My Passport/Data/JWST/ring_workshop')
##
plt.figure()
for ii in range(18):
    plt.subplot(3,9,ii+1)
    hdu = fits.open(path[ii+4])
    plt.imshow(level_adjust(hdu[1].data), cmap='gray', origin='lower')
    plt.title(path[ii+4][-29:-18])


##
for ii in range(18):
    hdu = fits.open(path[ii+4])
    print(hdu[0].header['filter'])
##
f335m = fits.open(path[2])[1].data.copy()
layers = mosaic(path[4:13], method='layers', plot=False)
f335med = mosaic(path[4:13], method='median', plot=False)
##
fac = 4
nanmean = np.zeros(f335med.shape)
for jj in range(f335med.shape[0]):
    for kk in range(f335med.shape[1]):
        vec = layers[jj, kk, :].copy()
        vec[vec > f335med[jj, kk] * (1+1/fac)] = np.nan
        vec[vec < f335med[jj, kk] * (1-1/fac)] = np.nan
        nanmean[jj, kk] = np.nanmean(vec)
    print(jj)
nanmean[np.isnan(nanmean)] = f335med[np.isnan(nanmean)]
##
c = 0
plt.figure()
for dat in [f335m, nanmean]:
    tit = ['mast', 'nanmean'][c]
    plt.subplot(1, 2, c+1)
    plt.imshow(level_adjust(dat, factor=1)[100: 900, 100:900], cmap='gray', origin='lower')
    plt.title(tit)
    c += 1

##

smooth = movmean(level_adjust(f335m.T, factor=1), win=40).T
clean = level_adjust(f335m, factor=1)
clean[clean < 0.1] = (clean-smooth)[clean < 0.1]
clean[clean < 0] = 0

c = 0
plt.figure()
for dat in [f335m, clean]:
    tit = ['mast', 'clean'][c]
    plt.subplot(1, 2, c+1)
    plt.imshow(level_adjust(dat, factor=1)[100: 900, 100:900], cmap='gray', origin='lower')
    plt.title(tit)
    c += 1
##
orig = level_adjust(f335m, factor=1)
hp = movmean(orig.T, win=40).T
ring = Ring2DKernel(15, 3)
lp = median_filter(orig, footprint=ring.array)
clean = lp + (orig - hp)
clean[clean < 0] = 0
clean[orig - clean > 0.1] = orig[orig - clean > 0.1]
# clean[clean < 0.1] = (clean-smooth)[clean < 0.1]
# clean[clean < 0] = 0
plt.imshow(clean, cmap='gray', origin='lower')
##
ring = Ring2DKernel(50, 3)
lp = median_filter(orig, footprint=ring.array)
hp = orig.copy()
for ii in range(hp.shape[0]):
    hp[ii, :] = medfilt(hp[ii, :], 101)
hp = orig-hp
clean = lp + hp
clean[clean < 0] = 0

c = 0
plt.figure()
for dat in [f335m, clean]:
    tit = ['mast', 'clean'][c]
    plt.subplot(1, 2, c+1)
    plt.imshow(level_adjust(dat, factor=1)[100: 900, 1100:1900], cmap='gray', origin='lower')
    plt.title(tit)
    c += 1

##
img = auto_plot('NGC6720', png='deband.png', pow=[1, 1, 1], pkl=False, resize=False, method='rrgggbb', plot=False,
          adj_args={'factor': 1}, max_color=False, fill=False, deband=True)

img = auto_plot('NGC6720', png=False, pow=[1, 1, 1], pkl=False, resize=False, method='rrgggbb', plot=False,
          adj_args={'factor': 2}, max_color=False, fill=False, deband=True)

bl = np.zeros(img.shape[0])
for ii in range(img.shape[0]):
    # bl[ii] = np.median(img[ii, 550:850])
    bl[ii] = np.percentile(img[ii, 500:2000], 50)
bls = bl - np.squeeze(movmean(bl, 201))
bls = movmean(bls, 7)
imgfix = img/255
for ii in range(img.shape[0]):
    imgfix[ii, :] = imgfix[ii, :] - bls[ii]/255 #'/ bls[ii] * np.median(bls)
imgfix[imgfix < 0] = 0
imgfix[imgfix > 1] = 1
plt.figure()
plt.imshow(imgfix)
plt.imsave('blc.png', imgfix, origin='lower')
    # plt.imshow(img[:, 550:850])
    # hp[ii, :] = medfilt(hp[ii, :], 101)

##
# smo9 = layers.copy()
# for lay in range(9):
#     clean = level_adjust(smo9[..., lay], factor=1)
#     smooth = movmean(clean.T, win=40).T
#     clean[clean < 0.1] = (clean - smooth)[clean < 0.1]
#     clean[clean < 0] = 0
#     smo9[..., lay] = clean
#
# c = 0
# plt.figure()
# for dat in [f335m, np.median(smo9, 2)]:
#     tit = ['mast', 'clean'][c]
#     plt.subplot(1, 2, c+1)
#     plt.imshow(level_adjust(dat, factor=1)[100: 900, 100:900], cmap='gray', origin='lower')
#     plt.title(tit)
#     c += 1

