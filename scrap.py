from astro_utils import *
# exp = ['jw06785-o001_t001_nircam_clear-f444w_i2d.fits', 'jw06785-o001_t001_nircam_clear-f335m_i2d.fits', 'jw06785-o001_t001_nircam_f444w-f470n_i2d.fits']
auto_plot('NGC-4298', exp = '*n_i2d.fits', png='rgb1n.jpg',adj_args={'factor':1},
          func=None, method='rrgggbb', fill=False, pkl=False, deband=False, deband_flip=False)

