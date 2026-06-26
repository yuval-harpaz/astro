from astro_utils import *


auto_plot('SMC-SW-Bar-3', exp='log', method='filt05', png='filt2all.jpg', crop=False, func=log1,
          adj_args={'factor':2}, fill=True, pkl=True, deband=False)
