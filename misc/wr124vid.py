import os
from matplotlib import pyplot as plt
from skimage import transform
import numpy as np
os.chdir('/home/innereye/JWST/WR124/')

big = plt.imread('WR124_0511.png')[:,:,:3]
center0 = [2250, 950]
center1 = [1888, 2420]
length = int(((2250-1888)**2+(2420-1950)**2)**0.5)
c0 = np.linspace(center0[0], center1[0], length).astype(int)
c1 = np.linspace(center0[1], center1[1], length).astype(int)
sq = np.linspace(750, 750/2, length).astype(int)
# centers = np.zeros(

# ii = 0
# sqh = sq[ii]
# cent = [c0[ii], c1[ii]]
# frame = big[cent[0]-sqh:cent[0]+sqh, cent[1]-sqh:cent[1]+sqh, :]
# frame = transform.resize(frame, (rs, rs))
# fig = plt.imshow(frame)
# plt.axis('off')
rs = 500
for ii in range(len(c0)):
    sqh = sq[ii]
    cent = [c0[ii], c1[ii]]
    frame = big[cent[0]-sqh:cent[0]+sqh, cent[1]-sqh:cent[1]+sqh, :]
    frame = transform.resize(frame, (rs, rs))
    plt.imsave('vid/wr124_'+str(ii).zfill(5)+'.png', frame)
    print(ii)
    # fig.set_array(frame)
    # plt.draw()
    # plt.pause(0.001)
while sqh < 1800:
    ii += 1
    sqh += 1
    frame = big[cent[0]-sqh:cent[0]+sqh, cent[1]-sqh:cent[1]+sqh, :]
    frame = transform.resize(frame, (rs, rs))
    plt.imsave('vid/wr124_' + str(ii).zfill(5)+'.png', frame)
    print(ii)
for ii in range(ii, ii+25):
    plt.imsave('vid/wr124_' + str(ii).zfill(5)+'.png', frame)
    print(ii)

# from astro_utils import *
# auto_plot('WR124', '*miri*_i2d.fits', png='WR_124_miri.png', pow=[1,1,1], factor=4, pkl=True, resize=True)
# auto_plot('WR124', '*nircam*_i2d.fits', png='WR_124_nircam.png', pow=[1,1,1], factor=4, pkl=True, resize=True)
# # from reproject import reproject_interp
# # import pickle
# import os
# from astro_fill_holes import *

# auto_plot('ngc5068', '*_i2d.fits', png=True, pow=[0.5, 1, 1], method='mnn', pkl=True, core=False)

path = np.asarray(list_files('/home/innereye/JWST/WR124/', '*.fits'))
filt = filt_num(path)
order = np.argsort(filt)
filt = filt[order]
path = path[order]
layers = np.load('WR124.pkl', allow_pickle=True)
# layers = np.load('WR124filled.pkl', allow_pickle=True)
# for ii in [6, 7, 8, 9]:
#     layers[:,:,ii] = np.roll(layers[:,:,ii], 23, axis=0)
#     layers[:,:,ii] = np.roll(layers[:,:,ii], -25, axis=1)
# plt.imshow(layers[:,:,[7,6,5]])

# layers = layers[475:4200, :, :]
# for ii in range(layers.shape[2]):
#     lay = layers[:,:,ii].copy()
#     xy = hole_xy(lay)
#     size = hole_size(lay, xy, plot=False)
#     lay = hole_disk_fill(lay, xy, size, larger_than=3)
#     mask = np.isnan(lay)
#     lay[mask] = 0
#     lay = level_adjust(lay)
#     lay[mask] = np.nan
#     layers[:,:,ii] = lay
#     print(ii)

# with open('WR124.pkl', 'wb') as f: pickle.dump(layers, f)
# bestx, besty, _ = optimize_xy_clust(layers[:,:,[0,3,8]], smooth=True, plot=True, neighborhood_size=None, thr=99.5)
# bestx, besty, layers = optimize_xy(layers, square_size=None, tests=9, plot=True)



auto_plot('WR124', '*_i2d.fits', png=True, pow=[0.5,1,1], factor=4, pkl=True, method='mnn', resize=True)

# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(layers[250:750,250:750,[0,1,3]])
# sq = 500
# xs = np.arange(0, layers.shape[0]-sq, sq)
# ys = np.arange(0, layers.shape[1]-sq, sq)
# shiftx = np.zeros((len(xs), len(ys), layers.shape[2]))
# shifty = np.zeros((len(xs), len(ys), layers.shape[2]))
# for xstart in xs:
#     for ystart in ys:
#         bestx, besty, _ = optimize_xy_clust(layers.copy()[xstart:xstart+500, ystart:ystart+500, :], smooth=False)
#         shiftx[int(xstart/500), int(ystart/500), :] = bestx
#         shifty[int(xstart/500), int(ystart/500), :] = besty
#         # print(f'YYY {ystart}')
#     print(f'XXXXXXX {xstart}')
#
# plt.subplot(1,2,2)
# plt.imshow(layers[250:750,250:750,[0,1,3]])

# with open('ngc5068.pkl', 'wb') as f: pickle.dump(layers, f)

# layers = layers[20:6000,300:4500,:]

#
# # crop = [3800,5000,5600,6800]
# margins = 100
# coord = [9700, 3500]
#
# layers = np.zeros((1080 + margins, 1920 + margins, len(path)))
# for ii in range(len(path)):
#     print('start :' + str(ii))
#     if ii == 0:
#         hdu0 = fits.open(path[ii])
#         # ref, ref_pos, ref_pix = crop_fits(hdu0[1], [6200, 4400], [1200, 1200])  # [4400, 6200]
#         orig = hdu0[1].copy()
#         ref, ref_pos, ref_pix = crop_fits(orig, coord, [1080+margins, 1920+margins])  # [4400, 6200]
#         img = ref.data
#         xy = hole_xy(img, x_stddev=6)
#         size = hole_size(img, xy, plot=False)
#         img = hole_disk_fill(img, xy, size, larger_than=2, allowed=0.5)
#         # img = hole_conv_fill(img, n_pixels_around=6, ringsize=15, clean_below=1)
#         hdr0 = ref.header
#         del hdu0
#     else:
#         hdu = fits.open(path[ii])
#
#         wcs = WCS(hdu[1].header)
#         pix = wcs.wcs_world2pix(ref_pos, 0)
#         pix = np.round(np.asarray(pix))
#         size = 2 * (pix[1, :] - pix[0, :])
#         hdu[1], _, _ = crop_fits(hdu[1], pix[1, :], size)
#         img = hdu[1].data
#         xy = hole_xy(img, x_stddev=6)
#         size = hole_size(img, xy, plot=False)
#         print('area = '+str(hdu[1].header['PIXAR_A2']))
#         print('prct 95 = '+str(np.round(np.percentile(np.nanmax(size, axis=1), 95), 1)))
#         img = hole_disk_fill(img, xy, size, larger_than=2, allowed=0.5)
#         # img = hole_conv_fill(img, n_pixels_around=3, ringsize=15, clean_below_local=0.75, clean_below=0.75)
#         # plt.figure();plt.imshow(img, origin='lower');plt.clim(0,1000);plt.show(block=False)
#         hdu[1].data = img
#         img, _ = reproject_interp(hdu[1], hdr0)
#         # img = img[crop[0]:crop[1],crop[2]:crop[3]]
#
#     if img.shape[0] == 0:
#         raise Exception('bad zero')
#     # img[img == 0] = np.nan
#     # med = np.nanmedian(img)
#     # img[np.isnan(img)] = 0
#     layers[:,:,ii] = img
#
#
# # layers = np.load('crop.pkl', allow_pickle=True)
# # layers = optimize_xy(layers)[2]
# ng = 2
# nudged = layers.copy()
# nudged[ng:,ng:,1] = nudged[:-ng,:-ng,1]
# nudged[ng:,ng:,0] = nudged[:-ng,:-ng,0]
# meds = 3.5
# # total = np.zeros(layers.shape[:2])
# rgbt = np.zeros((layers.shape[0], layers.shape[1], 3))
# c = 0
# b = []
# for ii in [0, 1, 2]:  # range(layers.shape[2]):
#     c += 1
#     img = nudged[:, :, ii]
#     img[img == 0] = np.nan
#     med = np.nanmedian(img)
#     img[np.isnan(img)] = 0
#     img = img - (med / meds)
#     if ii == 0:
#         img = img*2
#     elif ii == 2:
#         img = img*0.8
#     else:
#         img = img*1.3
#     img = img / (med * meds) * 255
#     img[img > 255] = 255
#     img[img < 0] = 0
#     # plt.subplot(2, 3, ii+1)
#     # plt.imshow(img, cmap='gray', origin='lower')
#     # plt.title(path[ii][-14:-9])
#     # plt.axis('off')
#     rgbt[..., 2-ii] = img
#
# rgbt = rgbt.astype('uint8')
# rgbt = rgbt[50:-50, 50:-50,:]
#
# plt.figure()
# plt.imshow(np.flipud(np.fliplr(rgbt)), origin='lower')
# plt.show()
#
#
# plt.imsave('capy_rgb.png', np.flipud(np.fliplr(rgbt)), origin='lower')
#
# ##
# # meds = 3
# total = np.zeros(layers.shape[:2])
# c = 0
# b = []
# for ii in [0,1,2]:  # range(layers.shape[2]):
#     c += 1
#     img = layers[:, :, ii]
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
# rgbt = rgbt[50:-50, 50:-50,:]
# plt.figure()
# plt.imshow(np.flipud(np.fliplr(rgbt)), origin='lower')
# plt.show()
# # plt.imsave('right_finger_tot.png', np.flipud(np.fliplr(rgbt)), origin='lower')
#
#
