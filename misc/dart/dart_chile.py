%matplotlib qt
from cv2 import circle
from astro_utils import *

## Take the first day data (before impact) to compute flat field

orig = plt.imread('/home/innereye/Data/DART/chile_27_sep.jpg')
gray = np.min(orig,2)
kernel = Gaussian2DKernel(x_stddev=5)
smoothed = convolve(gray, kernel)
xyc = np.unravel_index(smoothed.argmax(), smoothed.shape)
##
smooth = True  # False for raw, True for loading conv data, 2 for 9 pix square smoothing
zoom = False
clim1 = 1
max_rad = 755  # 145
height = 3
prom = 5
bl = np.percentile(gray, 45)
layer = gray - bl
tang_smooth = 10
if zoom:
    plot_trim = 75
else:
    plot_trim = 30

peaks = []
for rad in np.arange(2, max_rad, 1):
    img1 = circle(np.zeros((gray.shape[0], gray.shape[1], 3)), (xyc[1], xyc[0]), rad, (1, 0, 0), 1)[:, :, 0]
    x, y = np.where(img1[:, :xyc[1]])
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
    xc = np.asarray(list(x) + list(x1[1:]))
    yc = np.asarray(list(y) + list(xyc[1] - y[1:] + xyc[1] - 1))
    lc = smoothed[xc, yc]
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

#
plt.figure()
plt.imshow(smoothed, origin='lower', cmap='gray')
plt.clim(1,150)
for pk in peaks:
    plt.plot(pk[1],pk[0],'.r',markersize=1)

peaks = np.zeros(gray.shape)
mins = np.zeros(gray.shape)
for ii in range(gray.shape[0]):
    x = []
    x.append(np.arange(ii-10,ii+11))
    x.append(ii*np.ones(21, int))
    x.append(np.arange(ii-7,ii+8))
    x.append(x[-1])
    for jj in range(gray.shape[1]):
        y = []
        y.append(jj * np.ones(21, int))
        y.append(np.arange(jj-10,jj+11))
        y.append(np.arange(jj-7,jj+8))
        y.append(np.flipud(y[-1]))
        mn = 10000
        for kk in range(4):
            okay = np.ones(len(x[kk]), bool)
            okay[x[kk] < 0] = False
            okay[x[kk] >= gray.shape[0]] = False
            okay[y[kk] < 0] = False
            okay[y[kk] >= gray.shape[1]] = False
            dat = smoothed[x[kk][okay], y[kk][okay]]
            if smoothed[ii, jj] == np.max(dat):
                peaks[ii, jj] += 1
            if np.min(dat) < mn:
                mn = np.min(dat)
        mins[ii,jj] = mn
    print(ii)

##
select = peaks > 0
select[smoothed < 15] = False
select[smoothed / mins < 1.18] = False
# plt.figure()
# plt.imshow(select)

x, y = np.where(select)
plt.figure()
plt.imshow(gray, cmap='gray',origin='lower')
plt.plot(y,x,'.r',markersize=1)

