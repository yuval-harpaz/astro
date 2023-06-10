from astro_utils import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.stats import ttest_1samp

object = 'ngc 7319'
folder = list_files.__code__.co_filename[:-14]+'data/'+object.replace(' ', '_')
path3 = list_files(folder, search='*s3d.fits')
if len(path3) == 0:
# path = list_files('/home/innereye/JWST/Quintet/MAST_2022-08-25T1219/JWST/', search='*s3d.fits')
    include = 'jw02732-c1001_t004_miri_ch2-longmediumshort-'
    download_fits(object, extension='.fits', mrp=True, include=include, ptype='cube')

path1 = list_files('./', search='*x1d.fits')
path3 = list_files('./', search='*s3d.fits')  # run again to get inside subdir
# xy, size = mosaic_xy(path3, plot=False)
# max_xy = np.zeros((len(path3),2))
# for ii, fn in enumerate(path3):
hdu = fits.open(path3[0])
mx = np.zeros((hdu[1].data.shape[0],2))
for ff in range(hdu[1].data.shape[0]):
    mxi = np.argmax(hdu[1].data[ff,:,:])
    mx[ff,:] = np.unravel_index(mxi, (hdu[1].data.shape[1], hdu[1].data.shape[2]))
max_xy = [int(np.median(mx[:,0])),int(np.median(mx[:,1]))]

## clean the jumps
def remjum(vec, jumps, win=30):
    clean = vec.copy()
    for jump in jumps:
        clean[jump:] = clean[jump:] - \
            np.median(clean[jump:jump+win]) + \
            np.median(clean[jump-win:jump])
    return clean
jumps = [580, 626, 1249, 1320]
hdu1 = fits.open(path1[0])
tit = ['data from s3d.fits','removing jumps','smooth']
plt.figure()
for ip in [1,2,3]:
    plt.subplot(1,3,ip)
    for xx in range(21):
        jj = int(max_xy[0] - 11 + xx)
        vec = hdu[1].data[:, jj, int(max_xy[1])]
        if ip == 1:
            plt.plot(hdu1[1].data['WAVELENGTH'], vec)
        else:
            clean = remjum(vec, jumps)
            if ip == 2:
                plt.plot(hdu1[1].data['WAVELENGTH'], clean)
            else:
                plt.plot(hdu1[1].data['WAVELENGTH'], medfilt(clean, 101))
        plt.xlim(7.6, 11.6)
        plt.ylim(0,6000)
        plt.ylabel('MJy/sr')
        plt.xlabel('µm')
        plt.title(tit[ip-1])
plt.text(12,4000,'Close to AGN')
plt.text(12,500,'Away from AGN')
plt.text(8.5,4500,'Baseline')
plt.text(11,4500,'Baseline')
plt.text(9.7,4500,'Silicates\nAbsorption')
plt.show(block=False)


min_10 = np.zeros(hdu[1].data.shape[1:])
bl = min_10.copy()
dist = min_10.copy()
dist[:] = np.nan
bl_idx = []
for lim in [8.5, 9, 11, 11.5]:
    bl_idx.append(np.argmin(np.abs(hdu1[1].data['WAVELENGTH']-lim)))

for xx in range(21):
    jj = int(max_xy[0] - 11 + xx)
    for yy in range(21):
        kk = int(max_xy[1] - 11 + yy)
        vec = hdu[1].data[:, jj, kk]
        clean = remjum(vec, jumps)
        med = medfilt(clean, 101)
        bl[jj,kk] = (np.mean(med[bl_idx[0]:bl_idx[1]+1])+np.mean(med[bl_idx[2]:bl_idx[3]+1]))/2
        min_10[jj,kk] = np.min(med[bl_idx[1]+1:bl_idx[2]])
        dist[jj, kk] = ((jj - max_xy[0]) ** 2 + (kk - max_xy[1]) ** 2) ** 0.5

silicates = bl/min_10
silicates[np.isnan(silicates)] = 1
silicates[hdu[3].data[1000,:,:] == 513] = 100
plt.figure()
for ip in [1,2,3]:
    plt.subplot(1,3,ip)
    if ip == 1:
        plt.imshow(bl)
        plt.clim(0,3739)
        plt.title('brightness baseline (MJy/sr)')
    elif ip == 2:
        plt.imshow(min_10)
        plt.clim(0, 3739)
        plt.title('brightness minima around 10µm (MJy/sr)')
    else:
        plt.imshow(silicates)
        plt.clim(0.25, 4)
        plt.title('baseline/minimum')
    plt.axis('off')
    plt.colorbar()
    plt.ylim(9.5,29.5)
    plt.xlim(15.5,35.5)
plt.show(block=False)

dist_vec = dist.flatten()
sili_vec = silicates.flatten()
keep = (sili_vec > 0.25) & (sili_vec < 4) &  (~np.isnan(dist_vec))
dist_vec = dist_vec[keep]
sili_vec = sili_vec[keep]
dist_u = np.unique(dist_vec)
p = []
m = []
for u in dist_u:
    a = sili_vec[dist_vec == u]
    m.append(np.mean(a))
    _, stat = ttest_1samp(a, dist_u[0])
    p.append(stat)
m_med = m.copy()
m_med[2:] = medfilt(m_med[2:],3)
marker = np.asarray(p)
marker[marker > 0.05] = np.nan
marker[marker <= 0.05] = 0.75
plt.figure()
plt.plot(dist_vec,sili_vec,'.')
plt.plot([0,14],[1,1],'r:')
plt.plot(dist_u[:-1],m_med[:-1],'k--')
plt.ylim(0.25,4)
plt.plot(dist_u,marker,'c')
plt.title('Silicates absorption by distance from AGN')
plt.legend(['Absorption (baseline/10µm)', 'no absorption','mean absorption','sig one-sample t-test'])
plt.ylabel('absorption (ratio)')
plt.xlabel('distance from AGN (pixels)')
plt.show(block=False)



