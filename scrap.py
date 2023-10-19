import os

from astro_list_ngc import *


for hshift in range(-10, 10):
    h = 1053 + hshift
    for wshift in range(-10, 10):
        w = 633 + wshift
        h0 = layers.shape[0]-h
        w0 = layers.shape[1]-w
        subt = layers[h:,w:,:] - layers[:h0, :w0, :]
        img = np.zeros(subt.shape)
        for ii in range(3):
            img[..., ii] = level_adjust(subt[..., ii])
        plt.imsave(f'{w}_{h}.jpg', img[..., ::-1], origin='lower')

# ngc_html_thumb()
##
w = 632
h = 1051
h0 = layers.shape[0]-h
w0 = layers.shape[1]-w
subt = np.zeros((869, 867, 3))
for ii in range(3):
    notgreater = layers[:h0, :w0, ii].copy()
    lay = layers[h:, w:, ii]
    notgreater[notgreater > lay*1.2] = 0
    subt[..., ii] = lay - notgreater
img = np.zeros(subt.shape)
for ii in range(3):
    img[..., ii] = level_adjust(subt[..., ii])
plt.imsave('subt.jpg', img[..., ::-1], origin='lower')

##
os.chdir('/home/innereye/astro/data/IR23128/')
noise = glob('*o051*.fits')
data = glob('*o047*.fits')
for file in np.sort(data + noise):
    hdu = fits.open(file)
    print(hdu[1].shape)
tmp = noise[0]
noise[0] = noise[1]
noise[1] = tmp
##
img = np.zeros((561, 561, 3))
for ii in range(3):
    d = fits.open(data[ii])
    n = fits.open(noise[ii])
    img[..., 2-ii] = level_adjust(d[1].data[:561, :561] - n[1].data[:561, :561])
plt.imshow(img, origin='lower')

##
# os.chdir('/media/innereye/My Passport/Data/JWST/TRAPEZIUM-CLUSTER-P1/')
#
# auto_plot('TRAPEZIUM-CLUSTER-P1', exp='*clear*.fits', png='fac2.jpg', pkl=False, resize=True, method='rrgggbb',
#           plot=False, fill=True, deband=False, adj_args={'factor': 4}, blc=False, annotate=False, decimate=6)

# hdu = fits.open('jw01256-o001_t001_nircam_clear-f140m_i2d.fits')
# hdu[1].writeto('1_jw01256-o001_t001_nircam_clear-f140m_i2d.fits')



# download_obs()
# os.chdir('/media/innereye/My Passport/Data/JWST/NGC-6822-TILE-1')
# crop = 'y1=54; y2=3176; x1=2067; x2=7156'
# # img_file = 'filt_blc.png'
# # fits_file = 'jw01234-o010_t006_nircam_clear-f115w_i2d.fits'
# auto_plot('NGC-6822-TILE-1', exp='logNGC-6822-TILE-1.csv', png='bblc.png', pkl=True, resize=False, method='filt', plot=False,
#           crop=crop, fill=False, deband=False, adj_args={'factor': 1}, blc=True, annotate=1.2)
#
# os.chdir('/media/innereye/My Passport/Data/JWST/data/HH-30-MIRI/')
#
#
# def cpsf(img, center):
#     clean = np.zeros(img.shape)
#     for ph in range(center[0][-1], img.shape[0]):
#         for pw in range(center[1][-1], img.shape[1]):
#             point = [[ph, pw]]
#             fliph = center[0][0] - (point[0][0] - center[0][-1])
#             flipw = center[1][0] - (point[0][1] - center[1][-1])
#             if fliph > -1 and flipw > -1:
#                 point.append([fliph, point[0][1]])
#                 point.append([point[0][0], flipw])
#                 point.append([fliph, flipw])
#                 point = np.array(point)
#                 val = img[point[:, 0], point[:, 1]].copy()
#                 clean[point[:, 0], point[:, 1]] = val - np.min(val)
#     return clean
#
