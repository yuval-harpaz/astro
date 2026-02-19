from astro_utils import *
exp = ['jw05437-o003_t003_nircam_clear-f115w_i2d.fits',
 'jw05437-o003_t003_nircam_clear-f140m_i2d.fits',
 'jw05437-o003_t003_nircam_clear-f182m_i2d.fits',
 'jw05437-o003_t003_nircam_clear-f187n_i2d.fits',
 'jw05437-o003_t003_nircam_clear-f210m_i2d.fits',
 'jw05437-o003_t003_nircam_clear-f300m_i2d.fits',
 'jw05437-o003_t003_nircam_clear-f335m_i2d.fits',
 'jw05437-o003_t003_nircam_clear-f360m_i2d.fits',
 'jw05437-o003_t003_nircam_clear-f444w_i2d.fits',
 'jw05437-o003_t003_nircam_f150w2-f162m_i2d.fits',
 'jw05437-o003_t003_nircam_f405n-f444w_i2d.fits',
 'jw05437-o003_t003_nircam_f444w-f470n_i2d.fits']
auto_plot('W3-JWST-3A-FINAL', exp=exp[5:9], method='filt05', png='redder2.jpg', crop=False, func=None, adj_args={'factor':2}, fill=True, pkl=False)

