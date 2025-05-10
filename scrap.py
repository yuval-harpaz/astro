
from astro_utils import *
layers = auto_plot('WR112', exp='*.fits', method='filt05log', png='tmp.jpg', crop=False, func=log, adj_args={'factor':1}, fill=False, deband=False, deband_flip=None, pkl=False, opvar='layers')
for ii in range(3):
    layers[..., ii] = level_adjust(layers[..., ii], factor=1)

filt = np.array([770, 1500, 2100])
plt.figure()
for ii in range(2):
    for jj in range(3):
        rgb = assign_colors_by_filt(layers, filt, blc=ii, subtract_blue=[0, 0.5, 1][jj])
        plt.subplot(2, 3, jj+1 + 3*ii)
        plt.imshow(rgb, origin='lower')
        plt.axis('off')
        plt.title(f"blc={ii}, subtract_blue={[0, 0.5, 1][jj]}")


# if os.path.isdir(os.environ['HOME']+'/astro'):
#     os.chdir(os.environ['HOME']+'/astro')
#     if not os.path.isdir('data'):
#         os.mkdir('data')
# target = 'NGC-3132'
# logfile = glob('logs/'+target+'*')[0].split('/')[1]
# print('downloading')
# cwd = os.getcwd()
# time0 = time()
# download_by_log(logfile, tgt=target, overwrite=False, wget=False, path2data=cwd)
# print(f"finished in {int(np.round((time()-time0)/60))}min")
# os.chdir(cwd+'/data/')
# print('processing')
# time0 = time()
# auto_plot(target, exp='log'+logfile, method='filt', png='miri_filt.jpg',
#           adj_args={'factor':1}, fill=True, deband=True, deband_flip=True)
# print(f"finished in {int(np.round((time()-time0)/60))}min")
# print(os.getcwd())
# auto_plot(target, exp='log'+logfile, method='rrgggbb', png='miri_rgb.jpg',
#           adj_args={'factor':1}, fill=True, deband=True, deband_flip=True)
# print(f"finished in {int(np.round((time()-time0)/60))}min")
# print(os.getcwd())

# auto_plot('NGC-3132', exp='logNGC-3132.csv', method='rrgggbb', png='rgb1.jpg',
#           adj_args={'factor':1}, fill=True, deband=True, deband_flip=None, pkl=True)
# auto_plot('NGC-3132', exp='logNGC-3132.csv', method='filt', png='filt1.jpg',
#           adj_args={'factor':1}, fill=False, deband=False, deband_flip=None, pkl=True)
