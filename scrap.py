from astro_utils import *
os.chdir('/media/innereye/KINGSTON/JWST/data/NGC-3079')
files = glob('*.fits')
exp = ['jw05627-o019_t006_nircam_clear-f335m_i2d.fits', 'jw05627-o018_t006_miri_f560w-sub128_i2d.fits', 'jw05627-o018_t006_miri_f770w-sub128_i2d.fits']
auto_plot('NGC-3079', exp=exp, method='filt05', png='2.jpg', crop=False, func=None, adj_args={'factor':2}, fill=True, deband=False, deband_flip=True, pkl=False)
# plt.plot([1, 3, 1])
plt.figure()
for ii in range(4):
    print(ii)
    hdu = fits.open(files[ii])
    plt.subplot(2, 2, ii+1)
    plt.imshow(level_adjust(hdu[1].data))
    plt.title(files[ii].split('nircam')[1])
plt.show()
a = 1
