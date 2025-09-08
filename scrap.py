from astro_utils import *
# os.chdir('/media/innereye/KINGSTON/JWST/data/PN-CTR')
auto_plot('Ser-emb-11E', exp='*.fits', method='filt05', png='filt2db.jpg', crop=False,
           adj_args={'factor':2}, fill=True, deband=True, deband_flip=None, pkl=True, func=None)
