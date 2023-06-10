from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os

path = list_files('/home/innereye/JWST/ngc7469/',search='*.fits')
from astropy.convolution import Gaussian2DKernel, convolve
# path = list_files('ngc_628', search='*miri*.fits')
layers = mosaic(path,method='layers')


img560 = layers[:,:,1].copy()
img1 = layers[:,:,4]
idx = layers[:,:,1] == 0
img560[idx] = img1[idx]-6.5
plt.imshow(img560)
plt.clim(0, 10)
plt.show(block=False)

img770 = layers[:,:,2].copy()
img1 = layers[:,:,5]
idx = layers[:,:,2] == 0
img770[idx] = img1[idx]-10.5
plt.imshow(img770)
plt.clim(0, 10)
plt.show(block=False)

img1500 = layers[:,:,0].copy()
img1 = layers[:,:,3]
idx = layers[:,:,0] == 0
img1500[idx] = img1[idx]-59.5
plt.imshow(img1500)
plt.clim(0, 10)
plt.show(block=False)

rgb = np.zeros((img1.shape[0],img1.shape[1],3))
rgb[:,:,0] = img1500
rgb[:,:,1] = img770
rgb[:,:,2] = img560

final = rgb.copy()
# final = final**(1/10)
final[:,:,1] = final[:,:,1]/3
final = final/20*255
final[final > 255] = 255
final[final < 0] = 0
final = final.astype('uint8')
plt.figure()
plt.imshow(final)
# plt.clim(0, 200)
plt.show(block=False)
plt.imsave('both.png',final)

## spectra

path = list_files('/home/innereye/JWST/ngc7469/',search='*x1d.fits')
hdu = fits.open(path[1])
lims = [hdu[1].data['Wavelength'][0], hdu[1].data['Wavelength'][-1]]
table = get_lines(lims=lims)
z = 0.01627
expected = np.asarray(table['wavelength'])*(z+1)
atoms = np.asarray(table['species'])

plt.figure()
plt.plot(hdu[1].data['Wavelength'], hdu[1].data['FLUX'])
plt.title('Flux for NGC 7469 (AGN), x1d')
plt.ylabel(hdu[1].header['TUNIT2'])
plt.xlabel('Wavelength (µm)')
plt.grid()
plt.xticks(range(25))
for ii, line in enumerate(expected):
    plt.plot([line, line], [0, 0.85], 'k:')
    plt.text(line, 0.9, atoms[ii], rotation=90,ha='center')
plt.xlim(lims[0], lims[1])
plt.ylim(0, 1)
plt.show(block=False)

## cube
path1 = list_files('/home/innereye/JWST/ngc7469/',search='*1d.fits')
hdu1 = fits.open(path1[1])
z1 = evaluate_redshift(hdu1, max_z=0.2)
wavelength = hdu1[1].data['WAVELENGTH']
path = list_files('/home/innereye/JWST/ngc7469/',search='*3d.fits')
hdu = fits.open(path[1])
data = hdu[1].data
flower = np.log(np.mean(data, axis=0))
low = 5.4
flower[flower < low] = low
redshift = np.ones((data.shape[1],data.shape[1]))*z1
for xx in range(5, 40):
    for yy in range(5,40):
        if flower[xx,yy] > low*1.01:
            redshift[xx,yy] = evaluate_redshift(data[:, xx, yy], wavelength=wavelength, max_z=0.2)
    print(str(xx)+' of 40')


plt.figure()
plt.imshow(flower)
plt.show(block=False)

plt.figure()
plt.imshow(redshift)
plt.show(block=False)

from scipy.signal import medfilt
flux = hdu[1].data[:,19,19]
c = 299792458
freq = c/wavelength
y = flux[-1000:]
y = y - medfilt(y,31)
y = medfilt(y,3)
x = wavelength[-1000:]
xf = freq[-1000:]
pc = find_peaks(y,prominence=np.median(np.abs(y)))[0]

plt.figure()
plt.plot(x,flux[-1000:])
plt.plot(x,y)
plt.plot(x[pc],y[pc],'.')
plt.show(block=False)

cycle = -np.median(np.diff(xf[pc]))

# cycles = 2 # how many sine cycles
cycles = (freq[0] - freq[-1])/cycle
# cycles = 3
Fs = cycle
f = 1
sample = cycles*cycle
x = np.arange(sample)
y = np.sin(2 * np.pi * f * x / Fs)
plt.figure()
plt.plot(x, y)
plt.xlabel('sample(n)')
plt.ylabel('voltage(V)')
plt.show(block=False)

for ii in range(len(flux)):
    nearest = np.argmin(np.abs(freq[1]+x-freq[ii]))

plt.figure()
plt.plot(c/x*cycle, y)
plt.plot(wavelength,flux)
plt.show(block=False)