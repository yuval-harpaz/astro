from astro_utils import *
from astro_fill_holes import *
##
# auto_plot('Arp220', '*nircam*_i2d.fits', png=True, pow=[1,1,1], factor=4, pkl=True, method='rrgggbb', resize=True)
os.chdir('/home/innereye/JWST/Arp220')
layers = np.load('Arp220.pkl', allow_pickle=True)
for lay in range(layers.shape[2]):
    layers[:, :, lay] = level_adjust(layers[:, :, lay])
# xy = optimize_xy_manual(layers)
xy = np.zeros((layers.shape[2],2))
xy[6:,0] = -1
xy[6:,1] = 2
xy = xy.astype(int)
layers = np.load('Arp220.pkl', allow_pickle=True)
shifted = roll(layers, xy, nan_edge=True)
with open('Arp220.pkl', 'wb') as f:
    pickle.dump(shifted, f)
##
pow = [1, 1, 1]
auto_plot('Arp220',  '*miri*_i2d.fits', png='miri.png', pow=pow, pkl=True, resize=True, method='rrgggbb')
auto_plot('Arp220',  '*nircam*_i2d.fits', png='nircam.png', pow=pow, pkl=True, resize=True, method='rrgggbb')
method = 'rrgggbb'
auto_plot('Arp220',  '*_i2d.fits', png=method+'.png', pow=pow, plot=False, pkl=True, resize=True, method=method)
method = 'mnn'
auto_plot('Arp220',  '*_i2d.fits', png=method+'.png', pow=pow, plot=False, pkl=True, resize=True, method=method)
method = 'mtn'
auto_plot('Arp220',  '*_i2d.fits', png=method+'.png', pow=pow, plot=False, pkl=True, resize=True, method=method)

##
layers = layers[500:1000,600:1100,:]
with open('Arp220.pkl', 'wb') as f:
    pickle.dump(layers, f)
##