from astro_list_ngc import *

auto_plot('CB-26', exp = '*miri*.fits', method='rrgggbb', deband=10, deband_flip=True, png='miri_deband.jpg',adj_args={'factor':1}, func=None, fill=True, pkl=False)

