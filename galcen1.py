import pandas as pd

from astro_utils import *
from astro_fill_holes import *
# from time import time
os.chdir('/media/innereye/My Passport/Data/JWST/data/GALCEN')
log = pd.read_csv(os.environ['HOME'] + '/astro/logs/GALCEN_2022-09-19.csv')
log = log[log['file'] != 'jw01939-o001_t001_nircam_clear-f115w_i2d.fits']
# TODO: make log with fixed coord
crop = 'y1=206; y2=4487; x1=5710; x2=9938'
auto_plot('GALCEN', exp=log, png='cropped.png', pkl=False, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=True, deband=False, adj_args={'factor': 1}, crop=crop, blc=False, annotate=True)
auto_plot('GALCEN', exp=log, png='cropped2.png', pkl=False, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=True, deband=False, adj_args={'factor': 2}, crop=crop, blc=False)
rgb = auto_plot('GALCEN', exp=log, png='all.png', pkl=False, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=True, deband=False, adj_args={'factor': 1}, crop=False, blc=False)

rgb = plt.imread('all.png')
rgb = rgb[::-1, ...]

##
rgbc = rgb.copy()
rgbc[4:4454, :4650, 2] = rgbc[:4450, 4:4654, 2]
plt.imshow(rgbc, origin='lower')
plt.imsave('allc.png', rgbc, origin='lower')