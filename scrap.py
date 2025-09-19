from astro_utils import *

os.chdir('/media/innereye/KINGSTON/JWST/data/NAME-GAL-CENTER')
df = annotate_simbad('NAME-GAL-CENTER_NIRCam.jpg', 'jw04515-o101_t001_nircam_clear-f182m_i2d.fits', filter='NAME Galactic Circ')
df.to_csv('ann.csv', index=False)


