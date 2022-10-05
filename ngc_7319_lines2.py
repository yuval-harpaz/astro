from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
# from scipy.signal import medfilt, find_peaks
# import pandas as pd

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
redshift1d = []
for ii, fn in enumerate(path1):
    redshift1d.append(evaluate_redshift(yy[ii], wavelength=xx[ii], max_z=0.3, prom_med=20))
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

## 3D
path = list_files('./', search='*3d.fits')
hdu1 = fits.open(path1[1])
wavelength = hdu1[1].data['WAVELENGTH']
hdu = fits.open(path[0])
data = hdu[1].data
flower = np.log(np.mean(data, axis=0))

plt.figure()
plt.imshow(flower, origin='lower')
plt.show(block=False)
redshift = np.ones((data.shape[1],data.shape[2]))*0.022
for ix in range(9,30):
    for iy in range(16,37):
        # if flower[xx,yy] > low*1.01:
        redshift[ix,iy] = evaluate_redshift(data[:, ix, iy], wavelength=wavelength, max_z=0.2, prom_med=20)
    print(str(ix)+' x done')

sketch = redshift.copy()
sketch[sketch < 0.02] = 0.022
sketch[sketch > 0.024] = 0.022
sketch[18, 27] = 0  # dark spot
sketch[25, 24] = 0  # bright_spot
plt.figure()
plt.imshow(sketch, origin='lower',cmap='jet')
plt.clim(0.021, 0.023)
plt.show(block=False)

plt.figure()
c = 0
for ix in range(9,30,4):
    for iy in range(16,37,4):
        mx = 1180+np.argmax(data[1180:1200,ix,iy])
        c += 1
        plt.subplot(4,9,c)
        plt.plot(wavelength,data[:,ix,iy])
        plt.plot(wavelength[mx],data[mx,ix,iy],'.r')
        plt.text(wavelength[mx],data[mx,ix,iy],str(mx-1192))
        plt.title(str(ix)+','+str(iy))
print(c)
plt.show(block=False)

wlH2 = np.zeros((41,45))
for ix in range(41):
    for iy in range(45):
        mx = 1180+np.argmax(data[1180:1200,ix,iy])
        wlH2[ix,iy] = wavelength[mx]

plt.figure()
plt.imshow(wlH2, origin='lower',cmap='jet')
plt.clim(9.891, 9.903)
plt.show(block=False)

atomsU = np.unique(atoms)
material = np.zeros((41,45))
for ix in range(41):
    for iy in range(45):
        mx = np.argmax(data[:,ix,iy])
        if mx == 0:
            material[ix,iy] = np.nan
        else:
            material[ix,iy] = np.where(atomsU == atoms[np.argmin(np.abs(expected-wavelength[mx]))])[0][0]

board = material.copy()
board[-1,-1] = 25
board[-3,-1] = 13
board[-5,-1] = 49
board[-7,-1] = 40


plt.figure()
plt.imshow(board, origin='lower',cmap='jet')
# plt.clim(9.891, 9.903)
plt.text(45, 39.5, atomsU[25])
plt.text(45, 37.5, atomsU[13])
plt.text(45, 35.5, atomsU[49])
plt.text(45, 33.5, atomsU[40])
plt.show(block=False)

hdu[1].data[0,:,:] = material
fits.writeto('material.fits',hdu[1].data)
