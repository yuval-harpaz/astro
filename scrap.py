import os

from astro_list_ngc import *

auto_plot('OMC2-NE', exp = '*.fits', method='rrgggbb', deband=10, png='rgb1deband.jpg',adj_args={'factor':1}, func=log, fill=True, pkl=True)
