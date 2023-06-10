# from astro_utils import *
%matplotlib qt
from cv2 import circle
from astro_utils import *
# from astro_fill_holes import *
import os

## Take the first day data (before impact) to compute flat field
os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/')
stuff = '/home/innereye/astro/dart/'
path = list_files('/home/innereye/Dropbox/Moris_20220926-20221002/', '*.fits')
# kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data
mmdd = np.asarray([x[9:13] for x in path])
mmddu = np.unique(mmdd)
levs = 10.0**np.arange(-2,2.6,0.25)
levs = levs[1:]
tang_smooth = 10
## parameters and baseline correction
ver = '1'
gray = False
save = False
smooth = True  # False for raw, True for loading conv data, 2 for 9 pix square smoothing
zoom = False
clim1 = 1
height = 0.07

if smooth == True:
    data = np.load(stuff + 'per_day_smooth' + ver + '.pkl', allow_pickle=True)
    max_rad = 150  # 145
    prom = 0.005
else:
    data = np.load(stuff + 'per_day_raw' + ver + '.pkl', allow_pickle=True)
    max_rad = 212
    prom = 0.01
    height = 0.1
for day in range(7):
    layer = data[:, :, day]
    bl = np.percentile(layer, 45)
    layer = layer - bl
    if day == 4:
        layer = layer * 1.5
    if smooth == 2:  # smooth a little anyway
        for jj in range(1, 298):
            for kk in range(1, 298):
                layer[jj, kk] = np.mean(layer[jj - 1:jj + 2, kk - 1:kk + 2])
    data[:, :, day] = layer
## loop
if zoom:
    plot_trim = 75
else:
    plot_trim = 30
plt.figure()
for day in range(7):
    peaks = []
    for rad in np.arange(2, max_rad, 1):
        img1 = circle(np.zeros((300, 300, 3)), (150, 150), rad, (1, 0, 0), 1)[:, :, 0]
        x, y = np.where(img1[:, :151])
        idx = np.arange(len(x))
        xu = np.unique(x)
        for ii in range(int(len(xu) / 2)):
            xx = xu[ii]
            rows = np.where(x == xx)[0]
            if len(rows) > 1:
                yy = y[rows]
                order = np.argsort(-yy)
                idx[rows] = idx[rows[order]]
        y = y[idx]
        x1 = np.flipud(x)
        xc = np.asarray(list(x) + list(x1[1:-1]))
        yc = np.asarray(list(y) + list(300 - y[1:-1] - 1))
        lc = data[xc, yc, day]
        lc = np.asarray(list(lc) + list(lc[:10]))
        if smooth:  # already smoothed data
            dist = 10
            lcs = np.squeeze(movmean(lc, tang_smooth))
            # lcs = lc
        else:  # tangential smoothing
            dist = int(len(xc) / 8)
            lcs = np.squeeze(movmean(lc, tang_smooth))
        pks = find_peaks(lcs, distance=dist, prominence=prom, height=height)[0]  # , width=4
        pks = pks[pks > 5]
        pks[pks > len(xc) - 1] = pks[pks > len(xc) - 1] - len(xc)
        pks = np.unique(pks)
        for ipk in range(len(pks)):
            peaks.append([xc[pks[ipk]], yc[pks[ipk]]])
    d = data[:, :, day].copy()
    bl = (np.median(d[15:50, 15:50]) + np.median(d[-50:-15, -50:-15])) / 2
    d = d - bl
    d[d < levs[0]] = levs[0]
    plt.subplot(2, 4, day + 1)
    cs = plt.contourf(d, levs, norm=matplotlib.colors.LogNorm(), origin='lower', cmap='gray')
    plt.axis('off')
    plt.axis('square')
    plt.xlim(plot_trim, 300-plot_trim)
    plt.ylim(plot_trim, 300 - plot_trim)
    for pk in peaks:
        plt.plot(pk[1], pk[0],'.r',markersize=1)
plt.subplot(2, 4, 8)
plt.axis('off')
plt.colorbar(format='%.2f')

