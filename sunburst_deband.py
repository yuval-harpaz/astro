from astro_utils import *
os.chdir('/home/innereye/astro/data/sunburst/')
hdu0 = fits.open('jw02555-o003_t009_nircam_clear-f115w_i2d.fits')
# IR07251nircam.pkl
# plt.imshow(level_adjust(hdu0[1].data[2000:3000, 3000:4000], factor=1), origin='lower', cmap='gray')
data = hdu0[1].data[2000:3000, 3000:4000]
hdu0.close()

clean = data.copy()
clean = deband_layer(clean)
plt.imshow(level_adjust(clean, factor=1), origin='lower', cmap='gray')

auto_plot('sunburstfull', exp='log', png='rgb2deb10.png', pkl=True, resize=True, method='rrgggbb', blc=False,
          plot=False, fill=True, deband=True, adj_args={'factor': 2}, crop=False, whiten=False)