from astro_utils import *
os.chdir('/media/innereye/My Passport/Data/JWST/data/TRAPEZIUM-CLUSTER-P1/')

auto_plot('TRAPEZIUM-CLUSTER-P1', exp='*clear*.fits', png='fac2.jpg', pkl=False, resize=True, method='rrgggbb',
          plot=False, fill=True, deband=False, adj_args={'factor': 4}, blc=False, annotate=False, decimate=6)

##
img = plt.imread('trapezium.png')
for ii in range(3):
    img[..., ii] = hole_func_fill(img[..., ii])
plt.imsave('trap_fill.png', img)
orig = plt.imread('trapezium.png')
fix = img.copy()
fix[700:, :250,1] = orig[700:, :250,1]
fix[:150, 900:,1] = orig[:150, 900:,1]
fix[:180, :230,1] = orig[:180, :230,1]
fix[:, :40,0] = orig[:, :40,0]
fix[-40:, :,2] = orig[-40:, :,2]
plt.imsave('fix.png', fix)
# hdu = fits.open('jw01256-o001_t001_nircam_clear-f140m_i2d.fits')
# hdu[1].writeto('1_jw01256-o001_t001_nircam_clear-f140m_i2d.fits')

img = plt.imread('large.png')
img = img[282:1574, 10:2090,:]
for ii in range(3):
    img[..., ii] = hole_func_fill(img[..., ii])
plt.imsave('large_fill.png', img)
orig = plt.imread('large.png')
orig = orig[282:1574, 10:2090,:]
fix = img.copy()
fix[1170:, :250, 1] = orig[1170:, :250, 1]
fix[:100, :100, 1] = orig[:100, :100, 1]
plt.imsave('large_fix.png', fix)

img = plt.imread('proplydsDS9.png')
# img = img[282:1574, 10:2090,:]
for ii in range(3):
    img[..., ii] = hole_func_fill(img[..., ii])
plt.imsave('proplyds_filled.png', img)


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
