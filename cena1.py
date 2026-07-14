from astro_utils import *
# import pickle

os.chdir(drive+'data/Radio-Galaxy-NIRCam')
files = np.array(glob('*w_i2d.fits'))
filt = filt_num(files)
order = np.argsort(filt)
filt = filt[order]
files = files[order]


data = np.load('Radio-Galaxy-NIRCam.pkl', allow_pickle=True)
plt.figure()
for ii in range(9):
    plt.subplot(3, 3, ii+1)
    plt.imshow(level_adjust(data[..., ii+4]))
# with open('Radio-Galaxy-NIRCam.pkl', "rb") as f:
#     data = pickle.load(f)
#
# auto_plot('SMC-SW-Bar-3', exp='log', method='filt05', png='filt2all.jpg', crop=False, func=log1,
#           adj_args={'factor':2}, fill=True, pkl=True, deband=False)

# hdu0 = fits.open(files[3])
# img = hdu0[1].data[crop[0]:crop[1], crop[2]:crop[3]]
# # img = fill_holes(img, pad=1, hole_size=50)
# hdr0 = hdu0[1].header
# del hdu0
#
# for ii in range(4):
#     if ii == 0:
#
#     else:
#         hdu = fits.open(path[ii])
#         img, _ = reproject_interp(hdu[1], hdr0)
#         img = img[crop[0]:crop[1],crop[2]:crop[3]]
#     img[np.isnan(img)] = 0
#     img = img ** 0.5
#     img[img == 0] = np.nan
#     med = np.nanmedian(img)
#     img[np.isnan(img)] = 0
#     img = img - med
#     img = img / (med * meds - med) * 255
#
#     if img.shape[0] == 0:
#         raise Exception('bad zero')
#     # rgb[:,:,1] += img
#     total += img
#     img[img > 255] = 255
#     img[img < 0] = 0
#     if ii == 0:
#         rgb[:, :, 0] = img
# rgb[:, :, 2] = img
#
#     # with open('6layers.pkl', 'wb') as f:
#     #     pickle.dump(layers, f)
