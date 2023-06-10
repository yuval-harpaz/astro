# import matplotlib.pyplot as plt
from astro_utils import *
# from astro_fill_holes import *
# import cv2
# from scipy.ndimage import rotate
# from scipy.signal import medfilt
# from scipy.ndimage.filters import maximum_filter
# download_fits('SDSSJ1723+3411', extension='_i2d.fits', mrp=True, include='miri', ptype='image')
# from astroquery.mast import Observations
# obs_table = Observations.query_object('SDSSJ1723+3411')
# obs_table = obs_table[obs_table["dataRights"] == "PUBLIC"]
# obs_table = obs_table[obs_table["dataproduct_type"] == 'image']
# obs_table = obs_table[obs_table["obs_collection"] == "JWST"]
# all = Observations.get_product_list(obs_table)
# Observations.download_products(all[9])
##
# %matplotlib tk
##
os.chdir('/home/innereye/JWST/SDSSJ1723+3411')
auto_plot('SDSSJ1723+3411', '*nircam*_i2d.fits', png='smooth.png', pow=[1, 1, 1], method='rrgggbb', plot=False, smooth=True)
auto_plot('SDSSJ1723+3411', '*nircam*_i2d.fits', png='nosmooth.png', pow=[1, 1, 1], method='rrgggbb', plot=False)
auto_plot('SDSSJ1723+3411', '*_i2d.fits', png=False, pow=[1, 1, 1], method='mnn')
auto_plot('SDSSJ1723+3411', '*_i2d.fits', png=False, pow=[1, 1, 1], method='mnn', smooth=True)
layers = np.load('SDSSJ1723+3411.pkl', allow_pickle=True)
# with open('SDSSJ1723+3411_crop.pkl', 'wb') as f:
#     pickle.dump(layers, f)
for lay in range(layers.shape[2]):
    layers[:, :, lay] = level_adjust(layers[:, :, lay])
# with open('SDSSJ1723+3411_adjusted.pkl', 'wb') as f:
#     pickle.dump(layers, f)
plt.figure()
plt.imshow(layers[..., 0], origin='lower', cmap='gray')
plt.plot(layers[:, 480, 0] * 100 + 480, range(1000), 'r')
plt.plot(range(1000), layers[510, :, 0] * 100 + 510, 'c')
plt.axis('off')

##
os.chdir(os.environ['HOME']+'/JWST/SDSSJ1723+3411')
layers = np.load('SDSSJ1723+3411_adjusted.pkl', allow_pickle=True)