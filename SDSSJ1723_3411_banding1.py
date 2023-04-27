# import matplotlib.pyplot as plt
from astro_utils import *
# from astro_fill_holes import *
# import cv2
# from scipy.ndimage import rotate
# from scipy.signal import medfilt
# from scipy.ndimage.filters import maximum_filter
##
# %matplotlib tk
##
os.chdir('/home/innereye/JWST/SDSSJ1723+3411')
auto_plot('SDSSJ1723+3411', '*nircam*_i2d.fits', png='smooth.png', pow=[1, 1, 1], method='rrgggbb', plot=False, smooth=True)
auto_plot('SDSSJ1723+3411', '*nircam*_i2d.fits', png='nosmooth.png', pow=[1, 1, 1], method='rrgggbb', plot=False)


layers = np.load('SDSSJ1723+3411.pkl', allow_pickle=True)
with open('SDSSJ1723+3411_crop.pkl', 'wb') as f:
    pickle.dump(layers, f)