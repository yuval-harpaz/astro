from astro_list_ngc import *

auto_plot('CB-26', exp = 'log', method='filt', deband='nircam', deband_flip=False,
          png='both_filt6.jpg',adj_args={'factor':1}, func=None, fill=True, pkl=False, crop='y1=1930; y2=3496; x1=2074; x2=3663')

