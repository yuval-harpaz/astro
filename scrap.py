from astro_utils import *

os.chdir('/media/innereye/KINGSTON/JWST/data/NGC-2440')
auto_plot('NGC-2440', method='rrgggbb', deband=True, fill=True, pkl=True, png='rgb4_db_log.jpg', func=log1)
auto_plot('NGC-2440', method='filt05', deband=False, fill=False, pkl=True, png='filt4_db_log.jpg', func=log1)

