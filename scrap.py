from astro_utils import *
os.chdir('/media/yuval/KINGSTON/JWST/data/LRN_AT2018hso')

auto_plot('LRN_AT2018hso', exp='*fits', method='filt05', png='test.jpg', crop=False, func=log1, adj_args={'factor':1}, fill=True, pkl=False)
#
# files = glob('*.fits')
# plt.figure()
# for iff, ff in enumerate(files):
#     data = fits.open(ff)[1].data.copy()
#     data = hole_func_fill(data)
#     data = level_adjust(data)
#     plt.subplot(2,2, iff+1)
#     plt.imshow(data, origin='lower')
#     plt.title(ff)