# from astro_utils import *
%matplotlib qt
# from cv2 import circle
from astro_utils import *
from dart_tools import ridges
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
ver = '2'
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
    bl = np.nanpercentile(layer, 45)
    layer = layer - bl
    if day == 4:
        layer = layer * 1.36425565
    if smooth == 2:  # smooth a little anyway
        for jj in range(1, 298):
            for kk in range(1, 298):
                layer[jj, kk] = np.mean(layer[jj - 1:jj + 2, kk - 1:kk + 2])
    data[:, :, day] = layer
## loop
center = [302, 307]
if zoom:
    plot_trim = 75
else:
    plot_trim = 30
plt.figure()
for day in range(7):
    peaks = []
    for rad in np.arange(2, max_rad, 1):
        peaks = ridges(data[:, :, day], range(25, 191), tangential_smooth=10, prominence=0.01, height=0.04,
                       center=center, smooth=True, plot=False)
    d = data[:, :, day].copy()
    bl = (np.nanmedian(d[15:50, 15:50]) + np.median(d[-50:-15, -50:-15])) / 2
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

