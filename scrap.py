from astro_utils import *

auto_plot('LHA-120-N-159-NIRCam', exp='*nircam*.fits',method='filt05', resize=True, deband=False, fill=True, pkl=True, png='nircam_filt.jpg',func=log1)
print('done')
