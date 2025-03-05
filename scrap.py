from astro_utils import *

auto_plot('2MASS-J04302705+3545505', exp = 'log', method='rrgggbb', deband=False, deband_flip=False,
          png='rgb1nircam_crop_log.jpg',adj_args={'factor':1}, func=log, fill=True, pkl=True, crop='y1=2328; y2=3969; x1=2340; x2=4333')

