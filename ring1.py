from astro_utils import *

os.chdir('/media/innereye/My Passport/Data/JWST/UDS1or/')
path = glob('*/*.fits')+glob('*.fits')
path = np.sort(path)

for p in path:
    hdu = fits.open(p)
    print(p+' '+hdu[0].header['FILTER'])

data = mosaic(path, plot=False, method='mean')
data = level_adjust(data)
plt.imsave('mosaic.png', data, cmap='gray', origin='lower')

##
# PS1 261.42915 -22.0001

os.chdir('/media/innereye/My Passport/Data/JWST/data/LDN-57')
df = pd.read_csv('/home/innereye/astro/logs/LDN-57_2022-07-31.csv')
rows = np.where(df['file'].str.contains('f090w'))[0]

hdul = fits.open('PS1/rings.v3.skycell.0857.006.stk.g.unconv.fits')
hdul[1].header['CRVAL1'] = 261.42915
hdul[1].header['CRVAL2'] = -22.0001
hdul.writeto('PS1/ps1.fits', overwrite=True)
jw_ps = reproject(['jw01187-c1002_t003_nircam_clear-f090w_i2d.fits', 'PS1/ps1.fits'])
avg = (level_adjust(jw_ps[..., 0]) +level_adjust(jw_ps[..., 1]))/2
layers = mosaic(list(df.iloc[rows]['file']), method='layers', plot=False, log=df)
# for ii in range(len(rows)):
#     hdu = fits.open(df['file'][rows[rows[ii]]])
#     if ii == 0:
#         layers = np.


