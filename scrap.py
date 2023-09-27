from astro_utils import *
os.chdir('/media/innereye/My Passport/Data/JWST/NGC-6822-TILE-1')
crop = 'y1=54; y2=3176; x1=2067; x2=7156'
# img_file = 'filt_blc.png'
# fits_file = 'jw01234-o010_t006_nircam_clear-f115w_i2d.fits'
auto_plot('NGC-6822-TILE-1', exp='logNGC-6822-TILE-1.csv', png='bblc.png', pkl=True, resize=False, method='filt', plot=False,
          crop=crop, fill=False, deband=False, adj_args={'factor': 1}, blc=True, annotate=1.2)

