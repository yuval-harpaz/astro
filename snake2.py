from astro_utils import *
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u

os.chdir('/media/innereye/My Passport/Data/JWST/data/SNAKE-FIELD-2/')

##
%matplotlib qt
# 272.5305 -19.48789
##
result_table = Simbad.query_region(SkyCoord(ra=272.5305, dec=-19.48789,
                                   unit=(u.deg, u.deg), frame='fk5'),
                                   radius=0.1 * u.deg)

hdu = fits.open('jw01182-o002_t004_nircam_clear-f444w_i2d.fits')
wcs = WCS(hdu[1].header)
def add_time(spaced, h_d):
    toadd = [h_d, 'm', 's']
    spaced = spaced.split(' ')
    timed = ''
    for seg in range(len(spaced)):
        timed += spaced[seg]+toadd[seg]
    return timed
##
# wcs.wcs_world2pix(coo)
pix = np.zeros((len(result_table),2))
for ii in range(len(result_table)):
    ra = add_time(result_table[ii]['RA'], 'h')
    dec = add_time(result_table[ii]['DEC'], 'd')
    c = SkyCoord(ra=ra, dec=dec).to_pixel(wcs)
    pix[ii, :] = [c[0], c[1]]

##
inframe = (pix[:,0] > 0) & (pix[:,1] > 0) & (pix[:,0] <= hdu[1].shape[1]) & (pix[:,1] <= hdu[1].shape[0])
plt.figure()
plt.subplot(projection=wcs)
plt.imshow(level_adjust(hdu[1].data)**0.5, origin='lower')
plt.grid(color='white', ls='solid')
plt.xlabel('Galactic Longitude')
plt.ylabel('Galactic Latitude')
for idx in np.where(inframe)[0]:
    plt.plot(pix[idx, 0], pix[idx, 1], 'xr')
    plt.text(pix[idx, 0], pix[idx, 1], result_table[idx]['MAIN_ID'], color='k')
##
_ = auto_plot('SNAKE-FIELD-2', exp='*.fits', png='snake1.png', pkl=True, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=False, deband=False, adj_args={'factor': 1})

path = glob('*.fits')[1:]
##
_ = auto_plot('SNAKE-FIELD-2', exp=path, png='snake3_65.png', pkl=False, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=True, deband=False, adj_args={'factor': 1}, pow=[0.65, 0.65, 0.65])
# df = result_table.to_pandas()
##
_ = auto_plot('SNAKE-FIELD-2', exp='*.fits', png='snake1.png', pkl=True, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=False, deband=False, adj_args={'factor': 1}, crop=True)
## after fix coordinates create 3layer pkl
_ = auto_plot('SNAKE-FIELD-2', exp='logSNAKE-FIELD-2_no115.csv', png='snake3_65.png', pkl=True, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=True, deband=False, adj_args={'factor': 1}, pow=[0.65, 0.65, 0.65])
##
img = plt.imread('snake3_65.png')[..., :3]
for ii in range(3):
    layer = img[..., ii]
    layer[layer <= 0] = np.nan
    layer = layer - np.nanmin(layer)
    layer = layer / np.nanmax(layer)
    layer[np.isnan(layer)] = 0
    img[..., ii] = layer
plt.imsave('blc.png', img)
##  image with objects

os.chdir('/media/innereye/My Passport/Data/JWST/data/SNAKE-FIELD-2/')
img = plt.imread('blc.png')[..., :3]
img = img[::-1, ...]
hdu = fits.open('jw01182-o002_t004_nircam_clear-f200w_i2d.fits')
wcs = WCS(hdu[1].header)
result_table = Simbad.query_region(SkyCoord(ra=272.5305, dec=-19.48789,
                                   unit=(u.deg, u.deg), frame='fk5'),
                                   radius=0.1 * u.deg)
pix = np.zeros((len(result_table),2))
for ii in range(len(result_table)):
    ra = add_time(result_table[ii]['RA'], 'h')
    dec = add_time(result_table[ii]['DEC'], 'd')
    c = SkyCoord(ra=ra, dec=dec).to_pixel(wcs)
    pix[ii, :] = [c[0], c[1]]

inframe = (pix[:,0] > 0) & (pix[:,1] > 0) & (pix[:,0] <= hdu[1].shape[1]) & (pix[:,1] <= hdu[1].shape[0])
##
plt.figure()
# plt.subplot(projection=wcs)
plt.imshow(img, origin='lower')
# plt.imshow(level_adjust(hdu[1].data)**0.65)
# plt.grid(color='white', ls='solid')
for idx in np.where(inframe)[0]:
    # plt.plot(pix[idx, 0], pix[idx, 1], 'xr')
    plt.text(pix[idx, 0], pix[idx, 1], result_table[idx]['MAIN_ID'].replace('[WZT2014] ',''), color='r')
# plt.ylim(2000,3000)
# plt.xlim(2000,3000)
#
# df = result_table.to_pandas()

## snake-1
_ = auto_plot('SNAKE-FIELD-1', exp='logSNAKE-FIELD-1_no115.csv', png='snake3_65.png', pkl=True, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=True, deband=False, adj_args={'factor': 1}, pow=[0.65, 0.65, 0.65])
##
img = plt.imread('snake3_65.png')[..., :3]
for ii in range(3):
    layer = img[..., ii]
    layer[layer <= 0] = np.nan
    layer = layer - np.nanmin(layer)
    layer = layer / np.nanmax(layer)
    layer[np.isnan(layer)] = 0
    img[..., ii] = layer
plt.imsave('blc.png', img)
##  image with objects

os.chdir('/media/innereye/My Passport/Data/JWST/data/SNAKE-FIELD-1/')
img = plt.imread('blc.png')[..., :3]
img = img[::-1, ...]
hdu = fits.open('jw01182-o001_t001_nircam_clear-f200w_i2d.fits')
wcs = WCS(hdu[1].header)
result_table = Simbad.query_region(SkyCoord(ra=hdu[1].header['CRVAL1'], dec=hdu[1].header['CRVAL2'],
                                   unit=(u.deg, u.deg), frame='fk5'),
                                   radius=0.1 * u.deg)
pix = np.zeros((len(result_table),2))
for ii in range(len(result_table)):
    ra = add_time(result_table[ii]['RA'], 'h')
    dec = add_time(result_table[ii]['DEC'], 'd')
    c = SkyCoord(ra=ra, dec=dec).to_pixel(wcs)
    pix[ii, :] = [c[0], c[1]]

inframe = (pix[:,0] > 0) & (pix[:,1] > 0) & (pix[:,0] <= hdu[1].shape[1]) & (pix[:,1] <= hdu[1].shape[0])
##
plt.figure()
# plt.subplot(projection=wcs)
plt.imshow(img, origin='lower')
# plt.imshow(level_adjust(hdu[1].data)**0.65)
# plt.grid(color='white', ls='solid')
for idx in np.where(inframe)[0]:
    # plt.plot(pix[idx, 0], pix[idx, 1], 'xr')
    plt.text(pix[idx, 0], pix[idx, 1], result_table[idx]['MAIN_ID'].replace('[LCW2019] ','').replace('[WZT2014] ',''), color='r')
# plt.ylim(2000,3000)
# plt.xlim(2000,3000)