from astro_utils import *
# path = glob('*.fits')
os.chdir('/media/innereye/My Passport/Data/JWST/data/LDN-57')
df = pd.read_csv('/home/innereye/astro/logs/LDN-57_2022-07-31.csv')
rows = np.where(df['file'].str.contains('f090w'))[0]
##
if os.path.isfile('reproj90.pkl'):
    layers = np.load('reproj90.pkl', allow_pickle=True)
else:
    layers = reproject(list(df.iloc[rows]['file']), 4, log=df)
    with open('reproj90.pkl', 'wb') as f:
        pickle.dump(layers, f)

corrupt = layers[5456:, :, 3:5].copy()
layers[5456:, :, 3] = np.nan
layers[5456:, :, 4] = np.nan
#
# %matplotlib qt
# plt.figure()
# plt.imshow(layers[5000:6500, 5750:7500, [0, 0, 4]], origin='lower')
# toroll = layers[5456:, :, 3]
# toroll = np.roll(toroll, (63, -7), axis=(0, 1))
# # toroll = np.roll(toroll, 63, 1)
# layers[5456:, :, 3] = toroll
# plt.figure()
# plt.imshow(layers[5000:6500, 5750:7500, [0, 0, 3]], origin='lower')
##
avg = np.nanmedian(layers, 2)
idx = np.isnan(avg)
avg[idx] = np.min([corrupt[..., 0][idx[5456:, :]], corrupt[..., 1][idx[5456:, :]]], 0)
# hdu = fits.open('jw01187-c1002_t003_nircam_clear-f090w_i2d.fits')
# hdu[1].data = avg.copy()
# hdu[1].data[np.isnan(avg)] = 0
# hdu.writeto('f090w.fits')
avg = level_adjust(avg)
avg[np.isnan(avg)] = 0
plt.imsave('f090w.png', avg, origin='lower', cmap='gray')
## f356w, align to 090
pkl = 'reproj356.pkl'
rows = np.where(df['file'].str.contains('f356w'))[0]
if os.path.isfile(pkl):
    layers = np.load(pkl, allow_pickle=True)
else:
    layers = reproject(['jw01187-c1002_t003_nircam_clear-f090w_i2d.fits']+list(df.iloc[rows]['file']), 0, log=df)
    layers = layers[..., 1:]
    with open(pkl, 'wb') as f:
        pickle.dump(layers, f)
##
plt.figure()
plt.imshow(layers[5000:6500, 5750:7500, [0, 0, 3]], origin='lower')

layers[5456:, :, 3] = np.nan

avg = level_adjust(np.nanmedian(layers, 2))
avg[np.isnan(avg)] = 0
plt.imsave('f356w.png', avg, origin='lower', cmap='gray')
# layers[5456:, :, 4] = np.nan
##
red = plt.imread('f356w.png')[..., 0]
blue = plt.imread('f090w.png')[..., 0]
rgb = np.zeros((red.shape[0], red.shape[1], 3))
rgb[..., 0] = red
rgb[..., 1] = blue
rgb[..., 2] = blue

plt.imsave('rgb.png', rgb)

plt.figure()
plt.imshow(rgb)

#
# layers = mosaic(list(df.iloc[rows]['file']), method='layers', plot=False, log=df)
# avg = level_adjust(np.mean(layers, 2))
# plt.imsave('mosaic90.png', avg, origin='lower')
# hdul.writeto('PS1/ps1.fits', overwrite=True)
# jw_ps = reproject(['jw01187-c1002_t003_nircam_clear-f090w_i2d.fits', 'PS1/ps1.fits'])
# avg = (level_adjust(jw_ps[..., 0]) +level_adjust(jw_ps[..., 1]))/2

# for ii in range(len(rows)):
#     hdu = fits.open(df['file'][rows[rows[ii]]])
#     if ii == 0:
#         layers = np.

# hdul = fits.open('PS1/rings.v3.skycell.0857.006.stk.g.unconv.fits')
# hdul[1].header['CRVAL1'] = 261.42915
# hdul[1].header['CRVAL2'] = -22.0001
