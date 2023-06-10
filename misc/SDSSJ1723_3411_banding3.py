import numpy as np
import os
from matplotlib import pyplot as plt
from astro_utils import smooth_colors
from bot_grabber import level_adjust
%matplotlib tk
os.chdir(os.environ['HOME']+'/JWST/SDSSJ1723+3411')
layers = np.load('SDSSJ1723+3411.pkl', allow_pickle=True)
# plt.plot(layers[510, :, :6])

##
layers[np.isnan(layers)] = 0
for lay in range(layers.shape[2]):
    layers[..., lay] = layers[..., lay]
layers[..., :6] = smooth_colors(layers[..., :6])
layers[..., 6:] =  smooth_colors(layers[..., 6:])

# with open('SDSSJ1723+3411_smooth.pkl', 'wb') as f:
#     pickle.dump(layers, f)

##
from astro_utils import auto_plot
smc = auto_plot('SDSSJ1723+3411', '*nircam*_i2d.fits', png=False, pow=[1, 1, 1], method='rrgggbb', smooth=False,
          pkl='SDSSJ1723+3411_smooth.pkl', plot=False)
sm2 = auto_plot('SDSSJ1723+3411', '*nircam*_i2d.fits', png=False, pow=[1, 1, 1], method='rrgggbb', smooth=True,
          pkl='SDSSJ1723+3411_smooth.pkl', plot=False)
orig = auto_plot('SDSSJ1723+3411', '*nircam*_i2d.fits', png=False, pow=[1, 1, 1], method='rrgggbb', smooth=False,
          pkl='SDSSJ1723+3411_crop.pkl', plot=False)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(orig)
plt.axis('off')
plt.title('orig')
plt.subplot(1,3,2)
plt.imshow(smc)
plt.axis('off')
plt.title('remove extreme values')
plt.subplot(1,3,3)
plt.imshow(sm2)
plt.axis('off')
plt.title('medfilt')


## asaf
import os
from matplotlib import pyplot as plt
from astro_utils import smooth_colors
from bot_grabber import level_adjust
import numpy as np
from scipy.signal import find_peaks
%matplotlib tk
os.chdir(os.environ['HOME']+'/JWST/SDSSJ1723+3411')
layers = np.load('SDSSJ1723+3411_crop.pkl', allow_pickle=True)
layers[np.isnan(layers)] = 0
for lay in range(layers.shape[2]):
    layers[..., lay] = level_adjust(layers[..., lay])
# layers[..., :6] = smooth_colors(layers[..., :6])
# layers[..., 6:] =  smooth_colors(layers[..., 6:])



def find_block(img, smooth_par=5, peak_thresh=0.003, min_dist=50, plot=False):
    '''find vertical blocks with different bias in jwst NIRcam images
    img: image alligned with vertical blocks. NO NaN values
    smooth_par: size of running average to smooth after median filter
    peak_thresh: threshold for choosing peaks in derivative
    min_dist: minimum block size
    return: array of block boundaries'''
    # take the median over the columns
    med = np.median(img, axis=0)
    # smooth
    med = np.convolve(med, np.ones((smooth_par)), mode='same') / smooth_par
    # take the derivative
    deriv = np.convolve(med, np.array([-1, 0, 1]), mode='same')
    # find boundries of blocks
    p = find_peaks(abs(deriv), height=peak_thresh, distance=min_dist)
    bound, h = p
    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        ax[0].imshow(np.log1p(img), cmap='gray')
        ax[0].plot(range(deriv.shape[0]), -abs(deriv * 5000) + 500, 'r')
        ax[0].plot(bound, -h['peak_heights'] * 5000 + 500, 'bo', markersize=5)
        ax[1].plot(range(len(deriv)), abs(deriv))
        plt.show()
    return bound


def filt_block(img, bound, BG_level=0.4):
    ''' median filter on rows based on block boundaries.
    img: image alligned with vertical blocks. NO NaN values
    bound: array of block boundaries.
    BG_level: maximum level of background.
    return: filtered image.'''

    blocks = [(bound[i], bound[i + 1]) for i in range(len(bound) - 1)]
    blocks.append((bound[-1], img.shape[1]))
    blocks = [(0, bound[0])] + blocks
    # filter each row based on block
    filt_img = np.zeros(img.shape)
    for r in range(img.shape[0]):
        for start, finish in blocks:
            temp = img[r, start:finish]
            t = temp[temp < BG_level]
            q = np.median(t)
            filt_img[r, start:finish] = temp - q
    filt_img[filt_img < 0] = 0
    return filt_img


for lay in range(layers.shape[2]):
    bound = find_block(layers[:, :, lay], smooth_par=5, peak_thresh=0.003, min_dist=50)
    layers[..., lay] = filt_block(layers[..., lay], bound, BG_level=0.4)

# with open('SDSSJ1723+3411_bounds.pkl', 'wb') as f:
#     pickle.dump(layers, f)
from astro_utils import auto_plot
orig = auto_plot('SDSSJ1723+3411', '*nircam*_i2d.fits', png=False, pow=[1, 1, 1], method='rrgggbb', smooth=False,
          pkl='SDSSJ1723+3411_crop.pkl', plot=False)
sm2 = auto_plot('SDSSJ1723+3411', '*nircam*_i2d.fits', png=False, pow=[1, 1, 1], method='rrgggbb', smooth=False,
          pkl='SDSSJ1723+3411_smooth.pkl', plot=False)
bnd = auto_plot('SDSSJ1723+3411', '*nircam*_i2d.fits', png=False, pow=[1, 1, 1], method='rrgggbb', smooth=False,
          pkl='SDSSJ1723+3411_bounds.pkl', plot=False)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(orig)
plt.axis('off')
plt.title('orig')
plt.subplot(1,3,2)
plt.imshow(sm2)
plt.axis('off')
plt.title('remove extreme values')
plt.subplot(1,3,3)
plt.imshow(bnd)
plt.axis('off')
plt.title('bound')