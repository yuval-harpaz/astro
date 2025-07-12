from astro_utils import *
# auto_plot('SF_reg_1', exp='*.fits', method='filt05', png='filt2.jpg', crop=False, func=None, adj_args={'factor':2}, fill=True, deband=False, deband_flip=None, pkl=True)
# exp = ['jw06778-o001_t001_nircam_clear-f277w_i2d.fits', 'jw06778-o001_t001_nircam_clear-f335m_i2d.fits', 'jw06778-o001_t001_nircam_f444w-f470n_i2d.fits']
# auto_plot('SF_reg_1', exp=exp, method='rrgggbb', png='small2rgb.jpg', crop=False, func=None, adj_args={'factor':2}, fill=True, deband=False, deband_flip=None, pkl=False)
# exp = ['jw06778-o001_t001_nircam_clear-f090w_i2d.fits', 'jw06778-o001_t001_nircam_clear-f187n_i2d.fits', 'jw06778-o001_t001_nircam_clear-f200w_i2d.fits']
auto_plot('SF_reg_1', exp='*clear*.fits', method='filt05', png='reproj227.jpg', crop=False, func=None, adj_args={'factor':2}, fill=True, deband=False, deband_flip=None, pkl=True, reproject_to='f227w')

