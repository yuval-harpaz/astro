from astro_utils import *
os.chdir('/media/innereye/KINGSTON/JWST/SERPENS-MAIN')
auto_plot('SERPENS-MAIN', exp='*.fits', method='rrgggbb', png='rgb1.jpg', crop=False, func=None, adj_args={'factor':1}, fill=True, deband=False, deband_flip=None, pkl=True)

auto_plot('SERPENS-MAIN', exp='*.fits', method='rrgggbb', png='rgb1crop_log.jpg', crop="y1=4300; y2=5700; x1=750; x2=2750", func=log, adj_args={'factor':1}, fill=False, deband=False, deband_flip=None, pkl=True)

# from astro_utils import *
# layers = auto_plot('WR112', exp='*.fits', method='filt05log', png='tmp.jpg', crop=False, func=log, adj_args={'factor':1}, fill=False, deband=False, deband_flip=None, pkl=False, opvar='layers')
# for ii in range(3):
#     layers[..., ii] = level_adjust(layers[..., ii], factor=1)

# filt = np.array([770, 1500, 2100])
# plt.figure()
# for ii in range(2):
#     for jj in range(3):
#         rgb = assign_colors_by_filt(layers, filt, blc=ii, subtract_blue=[0, 0.5, 1][jj])
#         plt.subplot(2, 3, jj+1 + 3*ii)
#         plt.imshow(rgb, origin='lower')
#         plt.axis('off')
#         plt.title(f"blc={ii}, subtract_blue={[0, 0.5, 1][jj]}")



