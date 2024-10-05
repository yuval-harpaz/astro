from astro_utils import *
os.chdir('/media/innereye/KINGSTON/JWST/data/')
files = sorted(glob('NGC-1514*/*.fits'))
filt = filt_num(files)
files = np.array(files)[np.argsort(filt)]
bck = []
for f in files:
    if 'IMAGE-B' in f:
        bck.append(f)
        files = files[files != f]
folders = np.unique([f.split('/')[0] for f in files])
nfolders = len(folders)
filts = ['f770w', 'f1280w', 'f2550w']
bck_dict = dict(zip(filts, bck))
##
plt.figure()
for ii in range(len(folders)):
    img = np.zeros((1028, 1032, 3))
    fls = []
    for jj in range(3):
        fn = [f for f in files if filts[jj] in f and folders[ii] in f]
        hdu = fits.open(fn[0])
        img[..., 2-jj] = level_adjust(hdu[1].data)
    # print(hdu[1].data.shape)
    plt.subplot(3, 4, ii + 1)
    plt.imshow(img)
    plt.title(folders[ii])
imgb = np.zeros((1140, 1177, 3))
for jj in range(3):
    fn = bck[jj]
    hdub = fits.open(fn)
    imgb[..., 2-jj] = level_adjust(hdub[1].data)
# print(hdu[1].data.shape)
plt.subplot(3, 4, 12)
plt.imshow(imgb)
plt.title('bck')
##
save = False
bck_trim = []
for jj in range(3):
    fn = bck[jj]
    hdub = fits.open(fn)
    hdub = hdub[:2]
    hdub[1].data = hdub[1].data[100:1128, :1032]
    bck_trim.append(f'bck_{filts[jj]}.fits')
    if save:
        hdub.writeto(bck_trim[-1])
bck_dict = dict(zip(filts, bck_trim))

##
save = False
bck_trim = []
for jj in range(3):
    datamin = np.zeros((1028, 1032))
    datamin[...] = np.inf
    fn_freq = [f for f in files if filts[jj] in f]
    for fnf in fn_freq:
        hdu = fits.open(fnf)
        datamin = np.nanmin([datamin, hdu[1].data], 0)
    hdub = hdu.copy()[:2]
    hdub[1].data = datamin
    bck_trim.append(f'min_{filts[jj]}.fits')
    if save:
        hdub.writeto(bck_trim[-1])
bck_dict = dict(zip(filts, bck_trim))

# xy, size = mosaic_xy(files, plot=False)
# layers = mosaic(files, xy=xy, size=size, method='layers')
##
xy, size = mosaic_xy(files, plot=False)
layers = mosaic(files, xy=xy, size=size, method='layers', fill=True, subtract=bck_dict)
layers[layers == 0] = np.nan
##
data = np.zeros((layers.shape[0], layers.shape[1], 3))
for ii in range(3):
    data[..., ii] = np.nanmedian(layers[..., ii*nfolders:ii*nfolders+nfolders], 2)
##
datan = data.copy()
# datan[:800,:,np.array([0, 3, 6])+2] = np.nan
# datan[:800,685:,1] = np.nan
# datan[:800,:,4] = np.nan
img = np.zeros((datan.shape[0], datan.shape[1], 3))
for ii in range(3):
    img[..., 2-ii] = level_adjust(datan[..., ii])
# plt.figure()
# plt.imshow(img, origin='lower')
##
filt = [2550, 1280, 770]
col = matplotlib.cm.jet(filt / np.max(filt))[:, :3]  # [:, ::-1]
rgb = assign_colors(img, col)
for ic in range(3):
    rgb[:, :, ic] = rgb[:, :, ic] * 255
rgb = rgb.astype('uint8')
rgb = blc_image(rgb)
# plt.figure()
# plt.imshow(rgb, origin='lower')
plt.imsave('/home/innereye/Pictures/ngc1514min.jpg', rgb, origin='lower', pil_kwargs={'quality': 95})
##
layers = mosaic(files, xy=xy, size=size, method='layers', fill=False, subtract=bck_dict)
layers[layers == 0] = np.nan
data = np.zeros((layers.shape[0], layers.shape[1], 3))
for ii in range(3):
    data[..., ii] = np.nanmedian(layers[..., ii*nfolders:ii*nfolders+nfolders], 2)
datan = data.copy()
img = np.zeros((datan.shape[0], datan.shape[1], 3))
for ii in range(3):
    img[..., 2-ii] = level_adjust(datan[..., ii])
filt = [2550, 1280, 770]
col = matplotlib.cm.jet(filt / np.max(filt))[:, :3]  # [:, ::-1]
rgb = assign_colors(img, col)
for ic in range(3):
    rgb[:, :, ic] = rgb[:, :, ic] * 255
rgb = rgb.astype('uint8')
rgb = blc_image(rgb)
# plt.figure()
# plt.imshow(rgb, origin='lower')
plt.imsave('/home/innereye/Pictures/ngc1514nofill.jpg', rgb, origin='lower', pil_kwargs={'quality': 95})

##
# img = np.zeros((datan.shape[0], datan.shape[1], 3))
# for ii in range(3):
#     img[..., 2-ii] = level_adjust(datan[..., ii], factor=1)
# # plt.figure()
# # plt.imshow(img, origin='lower')
# filt = [2550, 1280, 770]
# col = matplotlib.cm.jet(filt / np.max(filt))[:, :3]  # [:, ::-1]
# rgb = assign_colors(img, col)
# for ic in range(3):
#     rgb[:, :, ic] = rgb[:, :, ic] * 255
# rgb = rgb.astype('uint8')
# rgb = blc_image(rgb)
# # plt.figure()
# # plt.imshow(rgb, origin='lower')
# plt.imsave('/home/innereye/Pictures/ngc1514_1.jpg', rgb, origin='lower', pil_kwargs={'quality': 95})
#
# ##
# # data = np.zeros((layers.shape[0], layers.shape[1], 3))
# # for ii in range(3):
# #     data[..., ii] = np.nanmean(layers[..., ii*3:ii*3+3], 2)
# # img = np.zeros(data.shape)
# # for ii in range(3):
# #     img[..., 2-ii] = level_adjust(data[..., ii])
# # plt.imshow(img, origin='lower')
# # ##
# # data = np.zeros(layers.shape)
# # for ii in range(9):
# #     data[..., ii] = level_adjust(layers[..., ii])
# # img = np.zeros((data.shape[0], data.shape[1], 3))
# # for ii in range(3):
# #     img[..., 2-ii] = np.nanmedian(data[..., ii*3:ii*3+3], 2)
# # plt.imshow(img, origin='lower')
# # plt.imsave('fac4.jpg', img, origin='lower', pil_kwargs={'quality':95})
# ##
# xy, size = mosaic_xy(files, plot=False)
# layers = mosaic(files, xy=xy, size=size, method='layers', fill=True)
# data = np.zeros(layers.shape)
# for ii in range(9):
#     data[..., ii] = level_adjust(log(layers[..., ii]))
# data[data == 0] = np.nan
# img = np.zeros((data.shape[0], data.shape[1], 3))
# for ii in range(3):
#     img[..., 2-ii] = np.nanmedian(data[..., ii*3:ii*3+3], 2)
# plt.imshow(img, origin='lower')
# plt.imsave('/home/innereye/Pictures/fill4log.jpg', img, origin='lower', pil_kwargs={'quality': 95})
# ##
# datan = data.copy()
# # datan[:800,:,np.array([0, 3, 6])+2] = np.nan
# datan[:800,685:,1] = np.nan
# datan[:800,:,4] = np.nan
# img = np.zeros((datan.shape[0], datan.shape[1], 3))
# for ii in range(3):
#     img[..., 2-ii] = np.nanmax(datan[..., ii*3:ii*3+3], 2)
# plt.figure()
# plt.imshow(img, origin='lower')
# ##
# filt = [2550, 1280, 770]
# col = matplotlib.cm.jet(filt / np.max(filt))[:, :3]  # [:, ::-1]
# rgb = assign_colors(img, col)
# for ic in range(3):
#     rgb[:, :, ic] = rgb[:, :, ic] * 255
# rgb = rgb.astype('uint8')
# rgb = blc_image(rgb)
# plt.figure()
# plt.imshow(rgb, origin='lower')
#
# ##
# # img = np.zeros((2110, 1777, 3))
