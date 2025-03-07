from astro_utils import *

auto_plot('NGC-5139', exp = '*nircam*.fits', deband=10,
          deband_flip=False, png='rgb1deband.jpg',adj_args={'factor':1}, func=None,
          fill=True, pkl=True)

