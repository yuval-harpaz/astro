from astro_utils import *
from astro_fill_holes import *
import os
import pickle
os.chdir('/media/innereye/My Passport/Data/JWST/data/')
path = list_files('VV-191','*_i2d.fits')
filt = filt_num(path)
order = np.argsort(filt)
path = np.asarray(path)[order]
# miri

layers = np.load('VV-191_full.pkl', allow_pickle=True)
layers = layers[200:2200,5800:7700,:]
with open('VV-191.pkl', 'wb') as f:
    pickle.dump(layers, f)
auto_plot('VV-191', exp='*.fits', png=True, pow=[1, 1, 1], pkl=True, resize=False, method='rrgggbb', plot=False)

make_thumb(plotted='VV-191.png', date0='2022-07-02')