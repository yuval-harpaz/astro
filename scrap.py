from astro_utils import *
# os.chdir('/media/innereye/KINGSTON/JWST/data/PN-CTR')
auto_plot('NGC-6537', exp='*.fits', method='filt05', png='filt1log_db.jpg', crop=False, func=log1,
           adj_args={'factor':1}, fill=True, deband=True, deband_flip=None, pkl=True)
