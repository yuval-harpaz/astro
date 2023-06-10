from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
# from scipy.signal import medfilt, find_peaks
# import pandas as pd
path = list_files('/home/innereye/JWST/SPT0418-47/MAST_2022-09-12T0535/JWST/',search='*3d.fits')
path1 = list_files('/home/innereye/JWST/SPT0418-47/MAST_2022-09-12T0535/JWST/',search='*1d.fits')
plt.figure()
for ii in range(8):
    hdu = fits.open(path[ii])
    avg = np.nanmean(hdu[1].data,axis=0)
    plt.subplot(2, 4, ii+1)
    plt.imshow(avg)
plt.show(block=False)

hdu[1].data.shape

for slice in [100,200,300,400,500,600]
    plt.subplot(2, 3, ii+1)
    plt.imshow(np.log10(avg))
object = 'ngc 7319'
root = list_files.__code__.co_filename[:-14]
table = get_lines()
folder = root+'data/'+object.replace(' ', '_')
ext = '*longmediumshort-_x1d.fits'
path1 = list_files(folder, search=ext)
path1 = list_files('./', search='*x1d.fits')  # shorter relative path than above
xx = []
yy = []
for ii, fn in enumerate(path1):
    hdu = fits.open(fn)
    xx.append(hdu[1].data['WAVELENGTH'])
    yy.append(hdu[1].data['FLUX'])


plt.figure()
for ii, fn in enumerate(path1):
    plt.plot(xx[ii], yy[ii], label=fn[-29:-26])
plt.title('Flux for NGC 7319 (AGN), extracted 1d')
plt.ylabel(hdu[1].header['TUNIT2'])
plt.xlabel('Wavelength (µm)')
plt.grid()
plt.xticks(range(25))
plt.xlim(2, 31)
plt.ylim(0, 0.6)
z = 0.022
expected = np.asarray(table['wavelength'])*(z+1)
atoms = np.asarray(table['species'])
for ii, line in enumerate(expected):
    plt.plot([line, line], [0, 0.2], 'k:')
    plt.text(line, 0.21, atoms[ii], rotation=90)
plt.show(block=False)
