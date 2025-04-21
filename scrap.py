from astro_utils import *
auto_plot('G286', exp ='log', method='rrgggbb',
          deband=True, deband_flip=False, png='clear_rgb1bd.jpg',
          adj_args={'factor':1}, func=None, fill=True, pkl=True, crop=False)
auto_plot('G286', exp ='log', method='filt',
          deband=False, deband_flip=False, png='clear_filt1bd.jpg',
          adj_args={'factor':1}, func=None, fill=False, pkl=True, crop=False)
