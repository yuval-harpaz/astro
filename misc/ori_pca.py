from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os
from reproject import reproject_interp
from sklearn import decomposition
path = list_files('/home/innereye/JWST/Ori/',search='*.fits')
from astropy.convolution import Gaussian2DKernel, convolve
# path = list_files('ngc_628', search='*miri*.fits')

## brief preview
if os.path.isfile('resamp.pkl'):
    img = np.load('resamp.pkl', allow_pickle=True)
else:
    plt.figure()
    for ii in range(len(path)):
        if ii == 0:
            hdu0 = fits.open(path[ii])
            img = hdu0[1].data[0:-1:8, 0:-1:6]
            img = np.zeros((img.shape[0],img.shape[1],len(path)))
            img[:,:,0] = hdu0[1].data[0:-1:8, 0:-1:6]
            hdr0 = hdu0[1].header
            del hdu0
        else:
            hdu = fits.open(path[ii])
            reproj, _ = reproject_interp(hdu[1], hdr0)
            img[:,:,ii] = reproj[0:-1:8, 0:-1:6]
        plt.subplot(3,5,ii+1)
        plt.imshow(img[:,:,ii])
        plt.title(path[ii][-20:])
        plt.show(block=False)
    with open('resamp.pkl', 'wb') as f:
        pickle.dump(img, f)

# layers = mosaic(path,method='layers')


plt.figure()
for ii in range(img.shape[2]):
    plt.subplot(3,5,ii+1)
    plt.imshow(img[:,1000:1630,ii], cmap='hot', origin='lower')
    slice = img[:,:,ii].flatten()
    slice = slice[slice>0]
    med = np.median(slice)
    plt.clim(med/4, med*4)
    plt.title(path[ii][-20:].replace('clear-','').replace('_i2d.fits',''))
    plt.axis('off')
# plt.clim(0, 200)
plt.show(block=False)
# plt.imsave('both.png',final)
mfilt = np.where(['m_i2d.fits' in x for x in path])[0][:-1]
slice = np.zeros((1110814, len(mfilt)))
for ii in range(len(mfilt)):
    slice[:, ii] = img[:, :, mfilt[ii]].flatten()


clean = slice[np.all(slice > 0, axis=1),:]
clean = clean[np.all(clean < 1000, axis=1),:]
clean = clean[np.all(~np.isnan(clean), axis=1),:]
plt.figure();plt.plot(clean);plt.show(block=False)

pca = decomposition.PCA(n_components=3)
# comp = pca.fit_transform(clean)

w = pca.fit(clean)
slice[np.isnan(slice)] = 0
slice[slice < 0] = 0
rgb = pca.transform(slice)
recon = np.reshape(rgb,(img.shape[0],img.shape[1],3))
clipped = recon/1000*250
clipped = clipped.astype('uint8')
plt.figure();plt.imshow(clipped);plt.show(block=False)
plt.figure()
for ii in range(3):
    plt.subplot(3,1,ii+1)
    plt.imshow(recon[:,:,ii])
    layer = recon[:, :, ii].flatten()
    layer = layer[layer > 0]
    med = np.median(layer)
    plt.clim(med / 4, med * 4)
    plt.axis('off')
plt.show(block=False)

plt.figure();plt.plot(w.components_.T);plt.show(block=False)

rescaled = recon.copy()[:,1000:1630,:]
order = [1,0,2]
for ii in [0,1,2]:
    layer = recon[:, 1000:1630, order[ii]]
    # layer = layer[layer > 0]
    med = np.median(layer)
    layer = layer-(med/4)
    layer = layer/(med*4)*255
    layer[layer > 255] = 255
    layer[layer < 0] = 0
    rescaled[:,:,ii] = layer

plt.figure()
plt.imshow(rescaled.astype('uint8'),origin='lower')
plt.show(block=False)

meds = 4
total = np.zeros((629,630))
for ii in range(len(mfilt)):
    # slice[:, ii] = img[:, :, mfilt[ii]].flatten()
    layer = img[:, 1000:1630, mfilt[ii]]
    # layer = layer[layer > 0]
    layer[np.isnan(layer)] = 0
    med = np.median(layer)
    layer = layer - (med / meds)
    layer = layer / (med * meds) * 255
    layer[layer > 255] = 255
    layer[layer < 0] = 0
    if ii == 0:
        r = layer
    elif ii == 5:
        b = layer
    total += layer
total = total/len(mfilt)
rgbt = np.zeros((629, 630, 3))
rgbt[..., 0] = b
rgbt[..., 1] = total
rgbt[..., 2] = r


plt.figure()
plt.imshow(rgbt.astype('uint8'), origin='lower')
plt.show(block=False)

