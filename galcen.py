from astro_utils import *
from astro_fill_holes import *
# from time import time
os.chdir('/media/innereye/My Passport/Data/JWST/data/GALCEN')
# red = np.load('GALCEN.pkl', allow_pickle=True)[..., -1]
##
hdu = fits.open('jw01939-o001_t001_nircam_f405n-f444w_i2d.fits')
img = hdu[1].data.copy()[1000:2000, 3000:4000]
# img = hole_func_fill(img, func='mean', fill_below=1)
img = hole_func_fill(img)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(hdu[1].data[1000:2000, 3000:4000])
plt.clim(0, 5000)
plt.subplot(1, 2, 2)
plt.imshow(img)
plt.clim(0, 5000)
##
auto_plot('GALCEN', exp='*.fits', png='fill2.png', pkl=True, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=True, deband=False, adj_args={'factor': 2})
##
path = glob('*.fits')[1:]
auto_plot('GALCEN', exp='*.fits', png='fill2.png', pkl=False, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=True, deband=False, adj_args={'factor': 1}, crop=True)

# TODO: make log with fixed coord
crop = 'y1=206; y2=4487; x1=5710; x2=9938'
auto_plot('GALCEN', exp=path, png='3layers.png', pkl=False, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=True, deband=False, adj_args={'factor': 1}, crop=crop, blc=False, annotate=True)

