from astro_utils import *
import os
os.chdir('/home/innereye/JWST/LDN57')
path = glob('*.fits')
data = mosaic(path, plot=False, method='mean')
data = level_adjust(data)
plt.imsave('mosaic.png', data, cmap='gray', origin='lower')