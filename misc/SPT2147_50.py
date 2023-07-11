from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np


# from scipy.signal import medfilt, find_peaks
# import pandas as pd
path = np.asarray(list_files('/media/innereye/My Passport/Data/JWST/SPT2147-50/',search='*i2d.fits'))
filt = filt_num(path)
path = path[np.argsort(filt)]
filt = np.sort(filt)
##
plt.figure()
for ii in range(15):
    plt.subplot(3,5,ii+1)
    hdu = fits.open(path[ii])
    layer = level_adjust(hdu[1].data)
    plt.imshow(layer, cmap='gray')
    plt.axis('off')
    plt.title(path[ii][:path[ii].index('/')]+'\n'+path[ii].replace('/','\n')[23:-5])
plt.show()
##

auto_plot('SPT2147-50', exp='*nircam*.fits', png='nircam.png' , pow=[1, 1, 1], pkl=False, resize=False, method='rrgggbb', plot=True)
auto_plot('SPT2147-50', exp='*nircam*.fits', png='nircam_crop.png' , pow=[1, 1, 1], pkl=False, resize=False, method='rrgggbb', plot=True, crop=True)

# crop = 'y1=1003; y2=2248; x1=949; x2=1926'
auto_plot('SPT2147-50', exp='*04_miri*.fits', png='miri2022_crop.png' , pow=[1, 1, 1], pkl=False, resize=False, method='rrgggbb', plot=True, crop=True)
crop = 'y1=298; y2=697; x1=667; x2=949',
exp = ['June2023/jw01355-o026_t024_miri_f560w_i2d.fits',
       'Sep2022/jw01355-o023_t004_miri_f770w_i2d.fits',
       'June2023/jw01355-o022_t024_miri_f1000w_i2d.fits',
       'Sep2022/jw01355-o023_t004_miri_f1280w_i2d.fits',
       'Sep2022/jw01355-o023_t004_miri_f1500w_i2d.fits',
       'Sep2022/jw01355-o023_t004_miri_f1800w_i2d.fits',
       'Sep2022/jw01355-o023_t004_miri_f2100w_i2d.fits']
auto_plot('SPT2147-50', exp=exp, png='miri20223_crop.png' , pow=[1, 1, 1], pkl=False, resize=False, method='rrgggbb', plot=True, crop=True)
exp = ['June2023/dif_f560w_i2d.fits',
       'Sep2022/jw01355-o023_t004_miri_f770w_i2d.fits',
       'June2023/dif_f1000w_i2d.fits',
       'Sep2022/jw01355-o023_t004_miri_f1280w_i2d.fits',
       'Sep2022/jw01355-o023_t004_miri_f1500w_i2d.fits',
       'Sep2022/jw01355-o023_t004_miri_f1800w_i2d.fits',
       'Sep2022/jw01355-o023_t004_miri_f2100w_i2d.fits']
auto_plot('SPT2147-50', exp=exp, png='miri20223dif_crop.png' , pow=[1, 1, 1], pkl=False, resize=False, method='rrgggbb', plot=True, crop=True)
##
exp = ['Sep2022/jw01355-o024_t004_nircam_clear-f200w_i2d.fits',
       'Sep2022/jw01355-o024_t004_nircam_clear-f277w_i2d.fits',
       'Sep2022/jw01355-o024_t004_nircam_clear-f356w_i2d.fits',
       'Sep2022/jw01355-o024_t004_nircam_clear-f444w_i2d.fits'
       'June2023/dif_miri_f560w_i2d.fits',
       'June2023/dif_miri_f1000w_i2d.fits']
# crop = 'y1=1003; y2=2248; x1=949; x2=1926'
auto_plot('SPT2147-50', exp='*04_miri*.fits', png='nircam2022miri2023_crop.png' , pow=[1, 1, 1], pkl=False, resize=False, method='rrgggbb', plot=True, crop=True)
# dif_f1000w.fits
##




a = ldict['a']
print(a)

for crp in crop.split(';'):
    cr = crp.split('=')
    for jj in [0, 1]:
        cr[jj] = cr[jj].strip()
    print(f'{cr[0]} {cr[1]}')
    locals()[cr[0]] = int(cr[1])