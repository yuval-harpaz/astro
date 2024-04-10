from astro_utils import *


os.chdir('/media/innereye/My Passport/Data/JWST/data/IR07251')
files = np.array([
       'jw03368-c1001_t003_miri_f1500w-brightsky_i2d.fits',
       'jw03368-c1001_t003_miri_f560w-brightsky_i2d.fits',
       'jw03368-c1001_t003_miri_f770w-brightsky_i2d.fits',
       'jw03368-o004_t003_miri_f1500w-brightsky_i2d.fits',
       'jw03368-o004_t003_miri_f560w-brightsky_i2d.fits',
       'jw03368-o004_t003_miri_f770w-brightsky_i2d.fits',
       'jw03368-o005_t003_miri_f1500w-brightsky_i2d.fits',
       'jw03368-o005_t003_miri_f560w-brightsky_i2d.fits',
       'jw03368-o005_t003_miri_f770w-brightsky_i2d.fits',
       'jw03368-o108_t003_nircam_clear-f150w_i2d.fits',
       'jw03368-o108_t003_nircam_clear-f200w_i2d.fits',
       'jw03368-o108_t003_nircam_clear-f277w_i2d.fits',
       'jw03368-o108_t003_nircam_clear-f356w_i2d.fits'])

pairs = [[4, 7], [5, 8], [3, 6]]
hdu0 = fits.open('jw03368-o108_t003_nircam_clear-f150w_i2d.fits')
# IR07251nircam.pkl
nircam = np.load('IR07251nircam.pkl', allow_pickle=True)
miri = np.zeros((1165, 1274, 3))
data = np.zeros((1165, 1274, 7))
data[..., :4] = nircam[561:1726, 718:1992, :]
del nircam


for pp in range(3):
    hdu = fits.open(files[pairs[pp][1]])
    hdun = fits.open(files[pairs[pp][0]])
    hdu[1].data = hdu[1].data - hdun[1].data[:561, :561]
    hdu[1].data[hdu[1].data < 0] = 0
    img, _ = reproject_interp(hdu[1], hdu0[1].header)
    data[..., pp+4] = img[561:1726, 718:1992]

with open('IR07251.pkl', 'wb') as f:
    pickle.dump(data, f)

auto_plot('IR07251', exp=files[6:], png='clean3.png', pkl=True, resize=False, method='mnn', blc=False,
          plot=False, fill=False, deband=False, adj_args={'factor': 3}, crop=False, whiten=False)

auto_plot('IR07251', exp=files[6:], png='clean3filt.png', pkl=True, resize=False, method='filt', blc=True,
          plot=False, fill=False, deband=False, adj_args={'factor': 3}, crop=False, whiten=False)
    # miri[..., pp] = level_adjust(img[561:1726, 718:1992], **{'factor': 3})
    # miri[..., pp] = level_adjust(hdu[1].data, **{'factor':3})
# plt.imshow(miri, origin='lower')
