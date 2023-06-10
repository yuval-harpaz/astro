import matplotlib.pyplot as plt
from astro_utils import *
from astro_fill_holes import *

os.chdir('/home/innereye/JWST/2MASS-09372521+0152514/')
path = list_files('/home/innereye/JWST/2MASS-09372521+0152514')
# layers = layers[1200:1600,1050:1450,:]
pow = [1, 1, 1]
auto_plot('2MASS-09372521+0152514', exp=path, png=True, pow=pow, pkl=True, resize=False, method='rrgggbb')
layers = np.load('2MASS-09372521+0152514.pkl', allow_pickle=True)
layers = layers[0:1100,400:1600,:]
for lay in range(3):
    layers[:, :, lay] = level_adjust(layers[:, :, lay])
    # layers[:, :, lay] = layers[:, :, lay] - np.min(layers[:, :, lay])
    # layers[:, :, lay] = layers[:, :, lay] / np.max(layers[:, :, lay])
# xy = optimize_xy_manual(layers)
conved = layers.copy()
for lay in range(3):
    conved[:, :, lay] = hole_conv_fill(conved[:, :, lay])
xy = optimize_xy_manual(conved)

plt.imshow(np.nanmedian(conved, 2)**1.5,origin='lower', cmap='hot')
# shifted = roll(conved, xy, nan_edge=True)
img = np.nanmedian(conved, 2)**1.5*255
img = img.astype('uint8')
plt.imsave('half_ring.png',img, cmap='hot',origin="lower")