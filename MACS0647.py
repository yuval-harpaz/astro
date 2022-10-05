from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import os
from reproject import reproject_interp
from sklearn import decomposition
import pickle


path = list_files('/home/innereye/JWST/MACS J0647.7+7015/',search='*.fits')[:-1]
crop = [40, 4340, 40, 4340]
rgb = np.zeros((crop[1]-crop[0], crop[3]-crop[2], 3), 'uint8')
total = np.zeros((crop[1]-crop[0], crop[3]-crop[2]), 'float')
meds = 4
subt = 50
# if os.path.isfile('6layers.pkl'):
#     layers = np.load('6layers.pkl', allow_pickle=True)
# else:
for ii in range(len(path)):
    if ii == 0:
        hdu0 = fits.open(path[ii])
        img = hdu0[1].data[crop[0]:crop[1],crop[2]:crop[3]]
        # img = fill_holes(img, pad=1, hole_size=50)
        hdr0 = hdu0[1].header
        del hdu0
    else:
        hdu = fits.open(path[ii])
        img, _ = reproject_interp(hdu[1], hdr0)
        img = img[crop[0]:crop[1],crop[2]:crop[3]]
    img[np.isnan(img)] = 0
    img = img ** 0.5
    img[img == 0] = np.nan
    med = np.nanmedian(img)
    img[np.isnan(img)] = 0
    img = img - med
    img = img / (med * meds - med) * 255

    if img.shape[0] == 0:
        raise Exception('bad zero')
    # rgb[:,:,1] += img
    total += img
    img[img > 255] = 255
    img[img < 0] = 0
    if ii == 0:
        rgb[:, :, 0] = img
rgb[:, :, 2] = img

    # with open('6layers.pkl', 'wb') as f:
    #     pickle.dump(layers, f)

total = total/len(path)
total[total > 255] = 255
total[total < 0] = 0

rgb[:, :, 1] = np.asarray(total, 'uint8')
plt.figure()
plt.imshow(rgb, origin='lower')
# plt.clim(20,200)
plt.show(block=False)
a = 1
#
# rgbt = np.zeros((total.shape[0], total.shape[1], 3))
# rgbt[..., 0] = np.mean(norm[:, :, :4]*1.5, axis=2)
# rgbt[..., 1] = total
# rgbt[..., 2] = np.mean(norm[:, :, 2:], axis=2)
# plt.figure()
# plt.imshow(rgbt.astype('uint8'), origin='lower')
# plt.show(block=False)
