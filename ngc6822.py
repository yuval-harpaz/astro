from astro_utils import *
os.chdir('/media/innereye/My Passport/Data/JWST/data')
path = glob('NGC-6822-NIRCAM-TILE-1/*nircam*.fits')
path = path + glob('NGC-6822-MIRI/*miri*.fits')
filt = filt_num(path)
path = np.array(path)[np.argsort(filt)]
##  'filt'
auto_plot('data', exp=path, png='NGC6822.png', pkl=False, resize=False, method='mnn', plot=False,
          crop=True, fill=True, deband=False, adj_args={'factor': 4})



## after aligning images in log, save pkl
os.chdir('/media/innereye/My Passport/Data/JWST/')
crop = 'y1=54; y2=3176; x1=2067; x2=7156'
auto_plot('NGC-6822-TILE-1', exp='logNGC-6822-TILE-1.csv', png='NGC6822.png', pkl=True, resize=False, method='mnn', plot=False,
          crop=crop, fill=True, deband=False, adj_args={'factor': 4})
##
auto_plot('NGC-6822-TILE-1', exp='logNGC-6822-TILE-1.csv', png='fac2.png', pkl=True, resize=False, method='mnn', plot=False,
          crop=crop, fill=False, deband=False, adj_args={'factor': 2})
##
auto_plot('NGC-6822-TILE-1', exp='logNGC-6822-TILE-1.csv', png='fac2w.png', pkl=True, resize=False, method='mnnw', plot=False,
          crop=crop, fill=False, deband=False, adj_args={'factor': 1})
##
auto_plot('NGC-6822-TILE-1', exp='logNGC-6822-TILE-1.csv', png='filt.png', pkl=True, resize=False, method='filt', plot=False,
          crop=crop, fill=False, deband=False, adj_args={'factor': 1})

## work on filt
from astro_utils import *
os.chdir('/media/innereye/My Passport/Data/JWST/NGC-6822-TILE-1')
img = plt.imread('filt.png')
img = blc_image(img)
plt.imsave('filt_blc.png', img)
img[..., 0] = np.max([img[..., 0], np.min(img[..., 1:], 2)], 0)
plt.imsave('filt_blc_w.png', img)
# path = list_files('/media/innereye/My Passport/Data/JWST/data/NGC-6822/')

##
os.chdir('/media/innereye/My Passport/Data/JWST/NGC-6822-TILE-1')
crop = 'y1=54; y2=3176; x1=2067; x2=7156'
img_file = 'filt_blc.png'
fits_file = 'jw01234-o010_t006_nircam_clear-f115w_i2d.fits'
annotate_simbad(img_file, fits_file, crop=crop, save=True)

##
header = fits.open(fits_file)[1].header

# else:  # TODO: don't flip for cv2, manage x y differently
#     img = img[::-1, ...]
wcs = WCS(header)
print('querying SIMBAD')
result_table = Simbad.query_region(
    SkyCoord(ra=header['CRVAL1'],
             dec=header['CRVAL2'],
             unit=(u.deg, u.deg), frame='fk5'),
    radius=0.1 * u.deg)
customSimbad = Simbad()
customSimbad.add_votable_fields('otype')
result_table1 = customSimbad.query_region(
    SkyCoord(ra=header['CRVAL1'],
             dec=header['CRVAL2'],
             unit=(u.deg, u.deg), frame='fk5'),
    radius=0.1 * u.deg)
