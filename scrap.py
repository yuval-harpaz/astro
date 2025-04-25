import os
pic = 'ngc_grid.png'
err = os.system(f"curl -T {pic} https://oshi.ec > tmp.txt")
if err:
    print('error sending to oshi')
else:
    with open('tmp.txt', 'r') as tmp:
        dest = tmp.read()
    download_link = dest.split('\n')[2].split(' ')[0]
    print(dest)
# from astro_utils import *

# from time import time


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
