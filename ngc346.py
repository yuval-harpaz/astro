from astro_utils import *
#
# log = pd.read_csv('logs/NGC-4321.csv')
# def crval_fix1(hd):
#     logrow = np.where(log['file'] == path[ii])[0]
#     if len(logrow) == 1:
#         for cr in [1, 2]:
#             correct = log.iloc[logrow][f'CRVAL{cr}fix'].to_numpy()[0]
#             if ~np.isnan(correct):
#                 hd[1].header[f'CRVAL{cr}'] = correct
#     return hd
pat = '/home/innereye/astro/data/NGC346/'
os.chdir(pat)
files = glob('*.fits')
filt = filt_num(files)
files = np.array(files)[np.argsort(filt)]
xy, size = mosaic_xy(files, plot=False)
layers = mosaic(files, xy=xy, size=size, method='layers')

data = np.zeros((layers.shape[0], layers.shape[1], 3))
for ii in range(3):
    data[..., ii] = np.nanmean(layers[..., ii*3:ii*3+3], 2)
img = np.zeros(data.shape)
for ii in range(3):
    img[..., 2-ii] = level_adjust(data[..., ii])
plt.imshow(img, origin='lower')
##
data = np.zeros(layers.shape)
for ii in range(9):
    data[..., ii] = level_adjust(layers[..., ii])
img = np.zeros((data.shape[0], data.shape[1], 3))
for ii in range(3):
    img[..., 2-ii] = np.nanmedian(data[..., ii*3:ii*3+3], 2)
plt.imshow(img, origin='lower')
plt.imsave('fac4.jpg', img, origin='lower', pil_kwargs={'quality':95})
##
xy, size = mosaic_xy(files, plot=False)
layers = mosaic(files, xy=xy, size=size, method='layers', fill=True)
data = np.zeros(layers.shape)
for ii in range(9):
    data[..., ii] = level_adjust(log(layers[..., ii]))
data[data == 0] = np.nan
img = np.zeros((data.shape[0], data.shape[1], 3))
for ii in range(3):
    img[..., 2-ii] = np.nanmedian(data[..., ii*3:ii*3+3], 2)
plt.imshow(img, origin='lower')
plt.imsave('fill4log.jpg', img, origin='lower', pil_kwargs={'quality': 95})



# img = np.zeros((2110, 1777, 3))
