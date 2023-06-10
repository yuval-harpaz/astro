
from astropy.io import fits
from astropy import wcs
from reproject import reproject_interp
import os
import glob
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from astropy.convolution import Ring2DKernel
from scipy.ndimage.filters import median_filter
from astroquery.mast import Observations

from astropy.coordinates import SkyCoord

hdul = fits.open('/home/innereye/astro/data/cartwheel/mastDownload/JWST/jw02727002001_02103_00002_nrcblong/jw02727002001_02103_00002_nrcblong_i2d.fits')
plt.imshow(hdul[1].data)
plt.clim(0.1,1)
plt.show(block=False)
coord = SkyCoord.from_pixel(1100, 1300, wcs.WCS(hdul[1].header))
coord.get_constellation()
astropy.coordinates.match_coordinates_sky(coord)
coord.match_coordinate_sky   ('mast')