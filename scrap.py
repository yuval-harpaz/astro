import os

from astro_utils import *
download_obs()
# os.chdir('/media/innereye/My Passport/Data/JWST/NGC-6822-TILE-1')
# crop = 'y1=54; y2=3176; x1=2067; x2=7156'
# # img_file = 'filt_blc.png'
# # fits_file = 'jw01234-o010_t006_nircam_clear-f115w_i2d.fits'
# auto_plot('NGC-6822-TILE-1', exp='logNGC-6822-TILE-1.csv', png='bblc.png', pkl=True, resize=False, method='filt', plot=False,
#           crop=crop, fill=False, deband=False, adj_args={'factor': 1}, blc=True, annotate=1.2)
#
os.chdir('/media/innereye/My Passport/Data/JWST/data/HH-30-MIRI/')


def cpsf(img, center):
    clean = np.zeros(img.shape)
    for ph in range(center[0][-1], img.shape[0]):
        for pw in range(center[1][-1], img.shape[1]):
            point = [[ph, pw]]
            fliph = center[0][0] - (point[0][0] - center[0][-1])
            flipw = center[1][0] - (point[0][1] - center[1][-1])
            if fliph > -1 and flipw > -1:
                point.append([fliph, point[0][1]])
                point.append([point[0][0], flipw])
                point.append([fliph, flipw])
                point = np.array(point)
                val = img[point[:, 0], point[:, 1]].copy()
                clean[point[:, 0], point[:, 1]] = val - np.min(val)
    return clean

