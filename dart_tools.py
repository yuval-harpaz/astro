# %matplotlib qt
from cv2 import circle
from astro_utils import *
from scipy.signal import find_peaks


def ridges(gray, rad_vec, tangential_smooth=10, prominence=None, height=None, center=None, smooth=False, plot=False):
    if center is None or smooth:
        kernel = Gaussian2DKernel(x_stddev=3)
        smoothed = convolve(gray, kernel)
        if center is None:
            center = np.unravel_index(smoothed.argmax(), smoothed.shape)
        if smooth:
            gray = smoothed
    if prominence is None:
        prominence = np.median(gray) * 0.75
    if height is None:
        height = prominence
    if type(rad_vec) == int:
        rad_vec = [rad_vec]
    peaks = []
    for rad in rad_vec:
        img1 = circle(np.zeros((gray.shape[0], gray.shape[1], 3)), (center[1], center[0]), rad, (1, 0, 0), 1)[:, :, 0]
        x, y = np.where(img1[:, :center[1]])
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
        yc = np.asarray(list(y) + list(center[1] - y[1:] + center[1] - 1))
        lc = gray[xc, yc]
        lc = np.asarray(list(lc) + list(lc[:10]))
        if smooth:  # already smoothed data
            dist = 10
            # lcs = np.squeeze(movmean(lc, tangential_smooth))
            # lcs = lc
        else:  # tangential smoothing
            dist = int(len(xc) / 8)
        lcs = np.squeeze(movmean(lc, tangential_smooth))
        pks = find_peaks(lcs, distance=dist, prominence=prominence, height=height)[0]  # , width=4
        pks = pks[pks > 5]
        pks[pks > len(xc) - 1] = pks[pks > len(xc) - 1] - len(xc)
        pks = np.unique(pks)
        for ipk in range(len(pks)):
            peaks.append([xc[pks[ipk]], yc[pks[ipk]]])
    if plot:
        tmp = gray# ** 0.5
        med = np.median(tmp)
        plt.imshow(tmp, origin='lower', cmap='gray')
        plt.clim(med/100, med*10)
        for pk in peaks:
            plt.plot(pk[1], pk[0], '.r', markersize=1)
    return peaks

# peaks = ridges(gray, range(2, 150), smooth=True, plot=True)

def sorty(xf,yf, div):
    idxf = np.arange(len(xf))
    xuf = np.unique(xf)
    if yf[-1] > yf[0]:
        sign = 1
    else:
        sign = -1
    for iif in range(int(len(xuf)/div)):
        xxf = xuf[iif]
        rowsf = np.where(xf == xxf)[0]
        if len(rowsf) > 1:
            yyf = yf[rowsf]
            orderf = np.argsort(sign * yyf)
            idxf[rowsf] = idxf[rowsf[orderf]]
    yf = yf[idxf]
    return(yf)
##
if __name__ == "__main__":
    data = np.load('/home/innereye/astro/dart/per_day_raw2.pkl', allow_pickle=True)
    gray = data[:, :, 1].copy()
    gray = gray - np.percentile(gray, 45)
    gray[gray < 0] = 0
    ridges(gray, rad_vec, tangential_smooth=10, prominence=None, height=None, center=None, smooth=False, plot=False)

# #
# plt.figure()
# plt.imshow(smoothed, origin='lower', cmap='gray')
# plt.clim(1,150)
# for pk in peaks:
#     plt.plot(pk[1],pk[0],'.r',markersize=1)
#
# peaks = np.zeros(gray.shape)
# mins = np.zeros(gray.shape)
# for ii in range(gray.shape[0]):
#     x = []
#     x.append(np.arange(ii-10,ii+11))
#     x.append(ii*np.ones(21, int))
#     x.append(np.arange(ii-7,ii+8))
#     x.append(x[-1])
#     for jj in range(gray.shape[1]):
#         y = []
#         y.append(jj * np.ones(21, int))
#         y.append(np.arange(jj-10,jj+11))
#         y.append(np.arange(jj-7,jj+8))
#         y.append(np.flipud(y[-1]))
#         mn = 10000
#         for kk in range(4):
#             okay = np.ones(len(x[kk]), bool)
#             okay[x[kk] < 0] = False
#             okay[x[kk] >= gray.shape[0]] = False
#             okay[y[kk] < 0] = False
#             okay[y[kk] >= gray.shape[1]] = False
#             dat = smoothed[x[kk][okay], y[kk][okay]]
#             if smoothed[ii, jj] == np.max(dat):
#                 peaks[ii, jj] += 1
#             if np.min(dat) < mn:
#                 mn = np.min(dat)
#         mins[ii,jj] = mn
#     print(ii)
#
# ##
# select = peaks > 0
# select[smoothed < 15] = False
# select[smoothed / mins < 1.18] = False
# # plt.figure()
# # plt.imshow(select)
#
# x, y = np.where(select)
# plt.figure()
# plt.imshow(gray, cmap='gray',origin='lower')
# plt.plot(y,x,'.r',markersize=1)
#
