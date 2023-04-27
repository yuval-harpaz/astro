import matplotlib.pyplot as plt
from astro_utils import *
from astro_fill_holes import *
import cv2
from scipy.ndimage import rotate
from scipy.signal import medfilt
# from scipy.ndimage.filters import maximum_filter
##
%matplotlib tk
##
os.chdir('/home/innereye/JWST/SDSSJ1723+3411')
img = np.load('f115w.pkl', allow_pickle=True)
img = level_adjust(img)
img_blur = cv2.GaussianBlur(img,(0,0), sigmaX=10, sigmaY=10)
##

##
# sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
gX = cv2.Sobel(img_blur, cv2.CV_64F, 1, 0)
gY = cv2.Sobel(img_blur, cv2.CV_64F, 0, 1)
# compute the gradient magnitude and orientation
magnitude = np.sqrt((gX ** 2) + (gY ** 2))
orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180
plt.imsave('tmp.png', orientation, cmap='gray')

##
os.chdir('/home/innereye/JWST/SDSSJ1723+3411')
img = np.load('f115w.pkl', allow_pickle=True)
img = level_adjust(img)
imgr = rotate(img, -1)
# plt.plot(medfilt(imgr[:,1000], 5))
# plt.plot(imgr[:,1000])
thr = np.nanpercentile(img, 20)
imgf = imgr.copy()
for ii in range(imgr.shape[0]):
    vec = medfilt(imgf[ii, :], 5)
    # vec[vec > thr] = imgr[ii, vec > thr]
    imgf[ii, :] = vec

for ii in range(imgr.shape[1]):
    vec = medfilt(imgf[:, ii], 5)
    # vec[vec > thr] = imgr[vec > thr, ii]
    imgf[:, ii] = vec

# imgf = imgf - np.nanpercentile(imgf, 5)
##
plt.figure()
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.clim(0,1)
plt.subplot(1,2,2)
plt.imshow(imgf, cmap='gray')
plt.clim(0,1)
##

os.chdir('/home/innereye/JWST/SDSSJ1723+3411')
img = np.load('f115w.pkl', allow_pickle=True)
img = level_adjust(img[:2000, :2000])

# ring = Ring2DKernel(9, 3)
# smooth_ring = median_filter(img, footprint=ring.array)
smooth = median_filter(img, footprint=np.ones((3, 3)))
imgf = img.copy()
for ii in range(img.shape[1]):
    vec = medfilt(imgf[:, ii], 5)
    # vec[vec > thr] = imgr[vec > thr, ii]
    imgf[:, ii] = vec
imgx = img.copy()
for ii in range(img.shape[0]):
    vec = medfilt(imgx[ii, :], 5)
    imgx[ii, :] = vec

imgyx = imgf.copy()
for ii in range(img.shape[0]):
    vec = medfilt(imgyx[ii, :], 5)
    imgyx[ii, :] = vec

imgxy = imgx.copy()
for ii in range(img.shape[1]):
    vec = medfilt(imgxy[:, ii], 5)
    # vec[vec > thr] = imgr[vec > thr, ii]
    imgxy[:, ii] = vec

plt.figure()
plt.subplot(2,3,1)
plt.imshow(img[700:1700, 700:1700], origin='lower')
plt.plot(img[700:1700,1200]*100+500, range(1000),'r')
plt.plot(range(1000), img[1300, 700:1700]*100+600,'k')
plt.title('orig')
plt.subplot(2,3,3)
plt.imshow(smooth[700:1700, 700:1700] , origin='lower')
plt.plot(smooth[700:1700,1200]*100+500, range(1000),'r')
plt.plot(range(1000), smooth[1300, 700:1700]*100+600,'k')
plt.title('smooth')
plt.subplot(2,3,4)
plt.imshow(imgf[700:1700, 700:1700] , origin='lower')
plt.plot(imgf[700:1700,1200]*100+500, range(1000),'r')
plt.plot(range(1000), imgf[1300, 700:1700]*100+600,'k')
plt.title('smooth col')
plt.subplot(2,3,5)
plt.imshow(imgx[700:1700, 700:1700] , origin='lower')
plt.plot(imgx[700:1700,1200]*100+500, range(1000),'r')
plt.plot(range(1000), imgx[1300, 700:1700]*100+600,'k')
plt.title('smooth row')
plt.subplot(2,3,6)
plt.imshow(imgyx[700:1700, 700:1700] , origin='lower')
plt.plot(imgyx[700:1700,1200]*100+500, range(1000),'r')
plt.plot(range(1000), imgyx[1300, 700:1700]*100+600,'k')
plt.title('smooth row')

plt.figure()
plt.subplot(2,3,1)
plt.imshow(smooth[700:1700, 700:1700] , origin='lower')
plt.plot(smooth[700:1700,1200]*100+500, range(1000),'r')
plt.plot(range(1000), smooth[1300, 700:1700]*100+600,'k')
plt.title('smooth')
plt.subplot(2,3,2)
plt.imshow(imgyx[700:1700, 700:1700] , origin='lower')
plt.plot(imgyx[700:1700,1200]*100+500, range(1000),'r')
plt.plot(range(1000), imgyx[1300, 700:1700]*100+600,'k')
plt.title('yx')
plt.subplot(2,3,3)
plt.imshow(imgxy[700:1700, 700:1700] , origin='lower')
plt.plot(imgxy[700:1700,1200]*100+500, range(1000),'r')
plt.plot(range(1000), imgxy[1300, 700:1700]*100+600,'k')
plt.title('xy')
## rotation
plt.figure()
c = 0
for rot in [0, 1]:
    if rot:
        tmp = rotate(img, -1)
    else:
        tmp = img.copy()
    for win in [3,5,7]:
        c += 1
        imgx = tmp.copy()
        for ii in range(img.shape[1]):
            vec = medfilt(imgx[:, ii], win)
            imgx[:, ii] = vec
        imgyx = imgx.copy()
        for ii in range(img.shape[0]):
            vec = medfilt(imgyx[ii, :], win)
            imgyx[ii, :] = vec
        imgy = tmp.copy()
        for ii in range(img.shape[0]):
            vec = medfilt(imgy[ii, :], win)
            imgy[ii, :] = vec
        img2 = np.array([imgx, imgy]).min(0)
        # plt.subplot(2, 3, c)
        # plt.imshow(imgyx[700:1700, 700:1700], origin='lower')
        # plt.plot(imgyx[700:1700, 1200] * 100 + 500, range(1000), 'r')
        # plt.plot(range(1000), imgyx[1300, 700:1700] * 100 + 600, 'k')
        plt.subplot(2, 3, c)
        plt.imshow(img2[700:1700, 700:1700], origin='lower')
        plt.plot(img2[700:1700, 1200] * 100 + 500, range(1000), 'r')
        plt.plot(range(1000), img2[1300, 700:1700] * 100 + 600, 'k')

## double smooth
plt.figure()
c = -1
cc = [1, 4, 2, 5, 3, 6]
for win in [3,5,7]:
    tmp = img.copy()
    for pas in [0, 1]:
        c += 1
        imgx = tmp.copy()
        for ii in range(img.shape[1]):
            vec = medfilt(imgx[:, ii], win)
            imgx[:, ii] = vec
        imgyx = imgx.copy()
        for ii in range(img.shape[0]):
            vec = medfilt(imgyx[ii, :], win)
            imgyx[ii, :] = vec
        imgy = tmp.copy()
        for ii in range(img.shape[0]):
            vec = medfilt(imgy[ii, :], win)
            imgy[ii, :] = vec
        img2 = np.array([imgx, imgy]).min(0)
        plt.subplot(2, 3, cc[c])
        plt.imshow(img2[700:1700, 700:1700], origin='lower', cmap='gray')
        plt.plot(img2[700:1700, 1200] * 100 + 500, range(1000), 'r')
        plt.plot(range(1000), img2[1300, 700:1700] * 100 + 600, 'c')
        plt.title(f'win={win}, x{pas+1}')
        plt.axis('off')
        tmp = img2


