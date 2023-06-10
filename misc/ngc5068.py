from astro_utils import *

from reproject import reproject_interp

import pickle
import os


# auto_plot('ngc5068', '*_i2d.fits', png=True, pow=[0.5, 1, 1], method='mnn', pkl=True, core=False)
os.chdir('/home/innereye/JWST/ngc5068/')
path = np.asarray(list_files('/home/innereye/JWST/ngc5068/', '*.fits'))
filt = filt_num(path)
order = np.argsort(filt)
filt = filt[order]
path = path[order]
layers = np.load('ngc5068.pkl', allow_pickle=True)
plt.figure()
plt.subplot(1,2,1)
plt.imshow(layers[250:750,250:750,[0,1,3]])
sq = 500
xs = np.arange(0, layers.shape[0]-sq, sq)
ys = np.arange(0, layers.shape[1]-sq, sq)
shiftx = np.zeros((len(xs), len(ys), layers.shape[2]))
shifty = np.zeros((len(xs), len(ys), layers.shape[2]))
for xstart in xs:
    for ystart in ys:
        bestx, besty, _ = optimize_xy_clust(layers.copy()[xstart:xstart+500, ystart:ystart+500, :], smooth=False)
        shiftx[int(xstart/500), int(ystart/500), :] = bestx
        shifty[int(xstart/500), int(ystart/500), :] = besty
        # print(f'YYY {ystart}')
    print(f'XXXXXXX {xstart}')

plt.subplot(1,2,2)
plt.imshow(layers[250:750,250:750,[0,1,3]])



## methods test
from astro_utils import *
from scipy import ndimage
os.chdir('/home/innereye/JWST/ngc5068/')
path = np.asarray(list_files('/home/innereye/JWST/ngc5068/', '*.fits'))
filt = filt_num(path)
order = np.argsort(filt)
filt = filt[order]
path = path[order]

plt.figure()
for sp, method in enumerate(['rrgggbb', 'mnn', 'mtn', 'filt']):
    img = auto_plot('ngc5068', '*_i2d.fits', png=False, pow=[1, 1, 1], pkl=True, resize=True, method=method, plot=False)
    plt.subplot(2,4, sp+1)
    plt.imshow(ndimage.rotate(img,90))
    plt.axis('off')
    plt.title(method)
for sp, method in enumerate(['rrgggbb', 'mnn', 'mtn', 'filt']):
    img = auto_plot('ngc5068', '*_i2d.fits', png=False, pow=[0.5, 1, 1], pkl=True, resize=True, method=method, plot=False)
    plt.subplot(2,4, sp+1+4)
    plt.imshow(ndimage.rotate(img,90))
    plt.axis('off')
    plt.title(method+', √red')
# with open('ngc5068.pkl', 'wb') as f: pickle.dump(layers, f)

