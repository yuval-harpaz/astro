from astro_utils import *
from astro_fill_holes import *
import os

os.chdir('/home/innereye/JWST/ngc1566/MAST_2022-12-01T1324/JWST/')
path = list_files('/home/innereye/JWST/ngc1566/MAST_2022-12-01T1324/JWST/','*_i2d.fits')
filt = filt_num(path)
order = np.argsort(filt)
path = np.asarray(path)[order]
# miri
miri = path[4:]
margins = 39
center = [770, 1000]
plt.figure()
for ii in range(4):
    hdu = fits.open(miri[ii])
    img = np.flipud(hdu[1].data.T)
    img = img[center[0]-int(1080/2):center[0]+int(1080/2), center[1]-int(1920/2):center[1]+int(1920/2)]
    plt.subplot(2,2,ii+1)
    plt.imshow(img)
    plt.clim(0,500)
plt.show(block=False)

layers = np.zeros((1080 + margins*2, 1920 + margins*2, len(path)))
for ii in range(len(miri)):
    print('start :' + str(ii))
    hdu = fits.open(miri[ii])
    img = np.flipud(hdu[1].data.T)
    img = img[center[0] - int(1080 / 2)-margins:center[0] + int(1080 / 2)+margins,
          center[1] - int(1920 / 2)-margins:center[1] + int(1920 / 2)+margins]
    orig = img.copy()
    img = hole_conv_fill(img, n_pixels_around=None, ringsize=10, clean_below=0)
    layers[:,:,ii] = img



# layers = np.load('crop.pkl', allow_pickle=True)
# layers = optimize_xy(layers)[2]
ng = [1, 5]
nudged = layers.copy()
# nudged[ng:,ng:,1] = nudged[:-ng,:-ng,1]
# nudged[ng[0]:,ng[1]:,0] = nudged[:-ng[0],:-ng[1],0]
nudged[:,ng[1]:,0] = nudged[:,:-ng[1],0]
nudged[:-ng[0],:,0] = nudged[ng[0]:,:,0]
mn = [5, 13, 21, 206]
mx = [56, 30, 77, 300]
cn = [635, 880]
w = 190
plt.figure()
for ii in range(4):
    plt.subplot(2,2,ii+1)
    plt.imshow(nudged[cn[0]-w:cn[0]+w,cn[1]-w:cn[1]+w,ii])
    plt.clim(mn[ii], mx[ii])
plt.show()

tmp3 = np.zeros((w*2,w*2,3),'uint8')
for ii in range(3):
    tmp = nudged[cn[0]-w:cn[0]+w,cn[1]-w:cn[1]+w,ii]
    tmp = tmp**(0.5)
    tmp = tmp-mn[ii]
    tmp = tmp/(mx[ii]-mn[ii])*255
    tmp = tmp.astype('uint8')
    tmp[tmp <= 0] = 0
    tmp[tmp >= 255] = 255
    tmp3[:,:,ii] = tmp
# plt.figure()
# plt.imshow(tmp3)
# plt.show()
meds = 3.5
# total = np.zeros(layers.shape[:2])
rgbt = np.zeros((layers.shape[0], layers.shape[1], 3), 'uint8')
b = []
for ii in [0, 1, 2]:  # range(layers.shape[2]):
    img = nudged[:, :, ii]
    img[img == 0] = np.nan
    # med = np.nanmedian(img)
    img[np.isnan(img)] = 0

    img = img - mn[ii]
    img = img / (mx[ii] - mn[ii]) * 255
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype('uint8')
    rgbt[..., 2-ii] = img
    del img
ng1 = [8, 8]
rgbt[:586-ng1[0],:1067,2] = rgbt[ng1[0]:586,:1067,2]
rgbt[:586,:1067-ng1[1],2] = rgbt[:586,ng1[1]:1067,2]
rgbt = rgbt[margins:-margins, margins:-margins,:]
plt.figure()
# plt.imshow(np.flipud(np.fliplr(rgbt)), origin='lower')
plt.imshow(rgbt, origin='lower')
plt.show()

gray = np.mean(rgbt, axis=2)
plt.figure()
# plt.imshow(np.flipud(np.fliplr(rgbt)), origin='lower')
plt.imshow(gray, origin='lower', cmap='gray')
plt.show()
plt.imsave('gray.png', np.flipud(np.fliplr(gray)), origin='lower', cmap='gray')
#
# ##
# # meds = 3
# total = np.zeros(layers.shape[:2])
# c = 0
# b = []
# for ii in [0,1,2]:  # range(layers.shape[2]):
#     c += 1
#     img = nudged[:, :, ii]
#     img[img == 0] = np.nan
#     med = np.nanmedian(img)
#     img[np.isnan(img)] = 0
#     img = img - (med / meds)
#     img = img / (med * meds) * 255
#     img[img > 255] = 255
#     img[img < 0] = 0
#     # plt.subplot(2, 3, ii+1)
#     # plt.imshow(img, cmap='gray', origin='lower')
#     # plt.title(path[ii][-14:-9])
#     # plt.axis('off')
#     if b == []:
#         b = img
#     total += img
# r = img
#
#
# total = total / c
# # layers = mosaic(path,method='layers')
# rgbt = np.zeros((total.shape[0], total.shape[1], 3))
# rgbt[..., 0] = r
# rgbt[..., 1] = total  # *3-r-b
# rgbt[..., 2] = b
#
# rgbt = rgbt.astype('uint8')
# rgbt = rgbt[margins:-margins, margins:-margins,:]
# plt.figure()
# plt.imshow(np.flipud(np.fliplr(rgbt)), origin='lower')
# plt.show()
# plt.imsave('right_finger_tot.png', np.flipud(np.fliplr(rgbt)), origin='lower')
#
#
