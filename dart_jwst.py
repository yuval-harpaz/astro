from astro_utils import *
import os
os.chdir('/home/innereye/JWST/DART/')
path = list_files('/home/innereye/JWST/DART/','*.fits')[:4]
for pp in path:
    hdu = fits.open(pp)
    print(hdu[0].header['DATE-OBS']+' '+hdu[0].header['TIME-OBS'])


tif = plt.imread('/home/innereye/JWST/DART/STSCI-H-p22047a-f-1320x440.tif')
