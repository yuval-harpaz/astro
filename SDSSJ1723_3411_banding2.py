import numpy as np
import os
from matplotlib import pyplot as plt
%matplotlib tk
# plt.figure()
# plt.imshow(layers[..., 0], origin='lower', cmap='gray')
# plt.plot(layers[:, 480, 0] * 100 + 480, range(1000), 'r')
# plt.plot(range(1000), layers[510, :, 0] * 100 + 510, 'c')
# plt.axis('off')

##
os.chdir(os.environ['HOME']+'/JWST/SDSSJ1723+3411')
layers = np.load('SDSSJ1723+3411_adjusted.pkl', allow_pickle=True)
# plt.plot(layers[510, :, :6])
vec = layers[510, :, :6].copy()
vec[np.isnan(vec)] = 0
vec = vec + 0.000000000001  # prevent nans
rank = np.argsort(-vec,1)
bad_rank = [vec[ii, rank[ii,0]]/vec[ii, rank[ii,1]] for ii in range(len(vec))]
bad_rank = np.array(bad_rank)
large_peak = np.ones(len(vec))
for ii in range(1, len(vec)-1):
    large_peak[ii] = vec[ii, rank[ii,0]]/np.max([vec[ii-1, rank[ii,0]]+vec[ii+1, rank[ii,0]]])


vecc = vec.copy()
for jj in np.where(bad_rank > 1.5)[0]:
    vecc[jj, rank[jj,0]] = (vec[jj-1, rank[jj,0]] + vec[jj+1, rank[jj,0]])/2

plt.figure()
plt.subplot(2,1,1)
plt.plot(vec)
# plt.xlim(270, 310)
plt.subplot(2,1,2)
plt.plot(vecc)
# plt.xlim(270, 310)

##
shift = 0.000000000001
img = layers[:,:,:6].copy()
# img = np.swapaxes(img, 0,1)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img[..., :3])
for row in range(img.shape[0]):
    vec = img[row, :, :6].copy()
    vec[np.isnan(vec)] = 0
    vec = vec + shift  # prevent nans
    rank = np.argsort(-vec, 1)
    bad_rank = [vec[ii, rank[ii, 0]] / vec[ii, rank[ii, 1]] for ii in range(len(vec))]
    bad_rank = np.array(bad_rank)
    large_peak = np.ones(len(vec))
    for ii in range(1, len(vec) - 1):
        large_peak[ii] = vec[ii, rank[ii, 0]] / np.max([vec[ii - 1, rank[ii, 0]] + vec[ii + 1, rank[ii, 0]]])
    vecc = vec.copy()
    for jj in np.where(bad_rank > 1.5)[0]:
        if jj < len(vec)-1:
            # vecc[jj, rank[jj, 0]] = (vec[jj - 1, rank[jj, 0]] + vec[jj + 1, rank[jj, 0]]) / 2
            # vecc[jj, rank[jj, 0]] = vec[jj - 1, rank[jj, 1]]
            vecc[jj, rank[jj, 0]] = np.median(vec[jj - 1, :])
    img[row,:,:] = vecc - shift
    print(row)
plt.subplot(1, 2, 2)
# img = np.swapaxes(img, 0,1)
plt.imshow(img[..., :3])

