from astro_deband import deband_layer
from time import time
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from astropy.io import fits
import os
import numpy as np
os.chdir('/home/innereye/astro/data/sunburst/')
hdu0 = fits.open('jw02555-o003_t009_nircam_clear-f115w_i2d.fits')
data = hdu0[1].data[3400:4400, 2900:3900]
hdu0.close()

##
funcs = [np.median, np.percentile, np.nanpercentile]
t1 = []
for ifunc in range(3):
    clean = data.copy()
    t0 = time()
    clean = deband_layer(clean, func=funcs[ifunc], prct=10)
    t1 .append(str(np.round(time()-t0, 1)))
    fnc = str(funcs[ifunc]).split(' ')[1]
    clean[clean > 1] = 1
    clean[np.isnan(clean)] = 1
    plt.imsave(fnc+'.png', clean, origin='lower', cmap='gray')
# t1 = ['4.2', '5.6', '190.1']
##
if not os.path.isfile('gmic.png'):
    data[data > 1] = 1
    plt.imsave('raw.png', data, origin='lower', cmap='gray')
    command = 'gmic raw.png -banding_denoise_v2 "10","0","10","20","1","0","0","50,50" -o gmic.png'
    t0 = time()
    os.system(command)
    t2 = str(np.round(time() - t0, 1))
else:
    t2 = '2.7'
gmic = plt.imread('gmic.png')
plt.subplot(2, 2, 1)
plt.imshow(gmic, cmap='gray')  # [50:-50, 50:-50]
plt.clim(0, 0.5)
fnc = 'gmic'
plt.title(fnc+' '+t1+'sec')
plt.show()
##

##
hdu0 = fits.open('jw02555-o003_t009_nircam_clear-f115w_i2d.fits')
data = hdu0[1].data[3400:4400, 2900:3900]
hdu0.close()
clean = data.copy()
t0 = time()
clean = deband_layer(clean, func=np.nanpercentile, win=401)
t3 = str(np.round((time()-t0), 1))
# plt.figure()
# plt.imshow(clean, cmap='gray', origin='lower')
# plt.clim(0, 0.5)
clean[clean > 1] = 1
plt.imsave(f'nanpercentile_win=401_{t3}sec.png', clean, origin='lower', cmap='gray')
##
images = ['gmic.png', 'median.png', 'percentile.png', 'nanpercentile.png']
plt.figure()
for iimg in range(4):
    img = plt.imread(images[iimg])
    plt.subplot(2,2,iimg+1)
    plt.imshow(img[..., 0], cmap='gray')
    plt.clim(0, 0.5)
    plt.title(images[iimg][:-4])


#
# def nanpercentile_worker(args):
#     data, q = args
#     return np.nanpercentile(data, q, axis=1)
#
#
# def parallel_nanpercentile(data, q, num_processes=8):
#     pool = mp.Pool(num_processes)
#     chunk_size = len(data) // num_processes
#     results = pool.map(nanpercentile_worker, [(data[i:i+chunk_size], q) for i in range(0, len(data), chunk_size)])
#     pool.close()
#     pool.join()
#     return np.concatenate(results)
#
#
# def smooth_width1(layer, win=101, prct=50):
#     half0 = int(win / 2)
#     half1 = win - half0
#     smoothed = layer.copy()
#     for ii in range(smoothed.shape[0]):
#         toavg = np.nan * np.ones((layer.shape[1] + win - 1, win))
#         # toavg = np.zeros((layer.shape[1] + win - 1, win))
#         for shift in np.arange(win):
#             toavg[shift:layer.shape[1] + shift, shift] = layer[ii, :]
#         # smoothed[ii, :] = nanpercentile(toavg, prct)[half0:-half1 + 1]
#         smoothed[ii, :] = parallel_nanpercentile(toavg, 10)[half0:-half1 + 1]
#         print(f'{ii}/{smoothed.shape[0]-1}', end='\r')
#     return smoothed
#
#
# def deband_layer1(layer, win=101, prct=10):
#     lp = smooth_width1(layer, win=win, prct=prct)
#     hp = layer - lp
#     lp = smooth_width1(lp.T, win=win, prct=prct).T
#     clean = lp + hp
#     clean[clean < 0] = 0
#     return clean

# Example usage
# data = np.random.rand(1000, 1000)  # Example 2D array
# q = 50  # Percentile value
# num_processes = 4  # Number of processes to use
#
# # result = parallel_nanpercentile(data, q, num_processes)
#
#
# os.chdir('/home/innereye/astro/data/IR09022/')
#
# # IR07251nircam.pkl
# # plt.imshow(level_adjust(hdu0[1].data[2000:3000, 3000:4000], factor=1), origin='lower', cmap='gray')
# ##
#
#
#
# def nanpercentile(arr, prct):
#     axis = 1
#     mask = ~np.isnan(arr)
#     count = mask.sum(axis=axis)
#     # groups = np.unique(count)
#     # groups = groups[groups > 0]
#     # percentile[
#     percentile = np.zeros((arr.shape[0]))
#     for g in range(len(groups)):
#         pos = np.where(count == groups[g])
#         values = arr[pos]
#         values = np.nan_to_num(values, nan=(np.nanmin(arr) - 1))
#         values = np.sort(values, axis=axis)
#         values = values[:, -groups[g]:]
#         percentile[pos] = np.percentile(values, prct, axis=axis)
#     return percentile
#
# def nanpercentile1(arr, prct):
#     axis = 1
#     mask = ~np.isnan(arr)
#     count = mask.sum(axis=axis)
#     groups = np.unique(count)
#     groups = groups[groups > 0]
#     percentile = np.zeros((arr.shape[0]))
#     for g in range(len(groups)):
#         pos = np.where(count == groups[g])
#         values = arr[pos]
#         values = np.nan_to_num(values, nan=(np.nanmin(arr) - 1))
#         values = np.sort(values, axis=axis)
#         values = values[:, -groups[g]:]
#         percentile[pos] = np.percentile(values, prct, axis=axis)
#     return percentile
#
# def smooth_width2(layer, win=101, prct=50):
#     half0 = int(win / 2)
#     half1 = win - half0
#     smoothed = layer.copy()
#     for ii in range(smoothed.shape[0]):
#         toavg = np.nan * np.ones((layer.shape[1] + win - 1, win))
#         # toavg = np.zeros((layer.shape[1] + win - 1, win))
#         for shift in np.arange(win):
#             toavg[shift:layer.shape[1] + shift, shift] = layer[ii, :]
#         smoothed[ii, :] = nanpercentile(toavg, prct)[half0:-half1 + 1]
#         # smoothed[ii, :] = np.nanmedian(toavg, axis=1)[half0:-half1 + 1]
#         print(f'{ii}/{smoothed.shape[0]-1}', end='\r')
#     return smoothed
#
# def smooth_width1(arr, prct=10, win=101):
#     till = int(win/2+0.5)
#     fro = till-win
#     result = np.zeros(arr.shape) * np.nan
#     for hh in range(arr.shape[0]):
#         row = np.zeros(arr.shape[1])
#         for ii in range(int(win/2), int(arr.shape[0]-win/2+0.5)):
#             # row[ii] = np.percentile(arr[hh, np.max([0, ii+fro]):np.min([len(arr)+1, ii+till])], prct)
#             row[ii] = np.percentile(arr[hh, ii + fro:ii + till], prct)
#         for ii in range(int(arr.shape[0]-win/2+0.5)):
#             row[ii] = np.percentile(arr[hh, np.max([0, ii+fro]):np.min([len(arr)+1, ii+till])], prct)
#             # row[ii] = np.percentile(arr[hh, ii + fro:ii + till], prct)
#         result[hh, :] = row
#     return result
#
#
# def deband_layer1(layer, win=101, prct=10):
#     lp = smooth_width1(layer, win=win, prct=prct)
#     hp = layer - lp
#     lp = smooth_width1(lp.T, win=win, prct=prct).T
#     clean = lp + hp
#     clean[clean < 0] = 0
#     return clean
#
# hdu0 = fits.open('jw03368-o113_t007_nircam_clear-f150w_i2d.fits')
# data = hdu0[1].data[3200:3600, 3200:3600]
# hdu0.close()
# clean = data.copy()
# t0 = time()
# clean = deband_layer1(clean)
# print(np.round(time()-t0))
# ##
# plt.figure()
# plt.imshow(level_adjust(clean, factor=1), origin='lower', cmap='gray')
# ##
#
