from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import medfilt
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
redshift = np.ones((data.shape[1],data.shape[2]))*z1
for xx in range(5, 40):
    for yy in range(5,40):
        if flower[xx,yy] > low*1.01:
            redshift[xx,yy] = evaluate_redshift(data[:, xx, yy], wavelength=wavelength, max_z=0.2, prom_med=20)
    print(str(xx)+' of 40')


plt.figure()
plt.imshow(flower)
plt.show(block=False)
#
plt.figure()
plt.imshow(redshift)
plt.clim(np.median(redshift)-0.0005, np.median(redshift)+0.0005)
plt.show(block=False)

flower_col = np.zeros((flower.shape[0],flower.shape[1],3))
red = redshift/np.median(redshift)
red[red>1.04] = 1
red[red<0.96] = 1
red = (red-1)*10+0.5
plt.figure()
plt.imshow(red)
# plt.clim(np.median(redshift)-0.0005, np.median(redshift)+0.0005)
plt.show(block=False)
r = red.copy()
r[r<0.5] = 0.5
b = red.copy()
b[r>0.5] = 0.5
# flower_col[:,:,0] = flower/flower.max()*r
# flower_col[:,:,1] = flower/flower.max()*0.5
# flower_col[:,:,2] = flower/flower.max()*b
flower_col[:,:,0] = flower/flower.max()*r
flower_col[:,:,1] = flower/flower.max()*0.5
flower_col[:,:,2] = flower/flower.max()*(1-b)
plt.figure()
plt.imshow(flower_col)
# plt.imshow(flower_col.astype('uint8'))
# plt.clim(np.median(redshift)-0.0005, np.median(redshift)+0.0005)
plt.show(block=False)



flux = hdu[1].data[:,19,19]
c = 299792458

freq = c/wavelength
cycle = 87323
fwin = 10**6/2
for samp in range(len(flux)):
    right = np.where(freq <= freq[samp]-fwin)[0]
    if len(right) > 0:
        right = right[0]
    else:
        right = int(samp)
    left = np.where(freq >= freq[samp]+fwin)[0]
    if len(left) > 0:
        left = left[-1]
    else:
        left = int(samp)
    y = flux[left:right]
    y = y-np.median(y)
    xl = wavelength[left:right]
    xf = freq[left:right]



    # cycles = 2 # how many sine cycles
    cycles = (freq[left] - freq[right]) / cycle + 1
    # cycles = 3
    Fs = cycle
    f = 1
    sample = (cycles + 1) * cycle
    x = np.arange(sample)
    sine = np.sin(2 * np.pi * f * x / Fs)

    idx = [mn,mx]
y = flux[-1000:]
y = y - medfilt(y, 31)
y = medfilt(y, 3)
xl = wavelength[-1000:]
xf = freq[-1000:]
pc = find_peaks(y, prominence=np.median(np.abs(y)))[0]


fluxmed = flux-medfilt(flux, 31)
start_segment = np.round(np.arange(freq[-1],freq[0],10**6/2))
ipeak = []
vpeak = []
for overlap = [0,1]:
    for seg in range(overlap,len(start_segment)2):
        idx = np.where(freq > start_segment[seg])[0][0]
# ymed = medfilt(y[pc], 5)
weight = np.zeros(len(flux))
for ii in range(len(flux)):
    idx = np.arange(-10,10)+ii
    idx = idx[idx >= 0]
    idx = idx[idx < len(flux)]
    weight[ii] = np.max(np.abs(fluxmed[idx]))
weight = medfilt(weight, 71)

plt.figure()
plt.plot(xl, flux[-1000:])
plt.plot(xl, y)
# plt.plot(xl[pc], ymed, '.')
plt.show(block=False)
plt.figure()
plt.plot(wavelength, fluxmed)
plt.plot(wavelength, weight)
plt.show(block=False)

cycle = -np.median(np.diff(xf[pc]))

# cycles = 2 # how many sine cycles
cycles = (freq[0] - freq[-1])/cycle
# cycles = 3
Fs = cycle
f = 1
sample = (cycles+1)*cycle
x = np.arange(sample)
sine = np.sin(2 * np.pi * f * x / Fs)
# phase = np.median(xf[pc] % cycle)/cycle
phase = 0.6
# for ii in range(len(flux)):
#     nearest = np.argmin(np.abs(freq[1]+x-freq[ii]))
nearest = np.round(freq-freq[-1]).astype(int)+int(np.round((1-phase)*cycle))
plt.figure()
plt.plot(wavelength, sine[nearest]*weight)
plt.plot(wavelength,flux)
plt.plot(wavelength,flux-sine[nearest]*weight)
plt.show(block=False)