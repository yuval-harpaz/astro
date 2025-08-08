from astro_utils import *
os.chdir('/media/innereye/KINGSTON/JWST/data/PN-CTR')
auto_plot('PN-CTR', exp='logPN-CTR_2024-07-23.csv', method='filt05', png='filt1crop_db.jpg', crop='y1=30; y2=4922; x1=13; x2=4834', func=None,
           adj_args={'factor':1}, fill=False, deband=False, deband_flip=None, pkl=True)
