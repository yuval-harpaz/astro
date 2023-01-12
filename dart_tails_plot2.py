%matplotlib qt
##
from astro_utils import *
from astro_fill_holes import *
from dart_tools import ridges
import os
from cv2 import putText, FONT_HERSHEY_DUPLEX


## Take the first day data (before impact) to compute flat field
os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/')
stuff = '/home/innereye/astro/dart/'
path = list_files('/home/innereye/Dropbox/Moris_20220926-20221002/', '*.fits')
# kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data
mmdd = np.asarray([x[9:13] for x in path])
mmddu = np.unique(mmdd)
pix_size = [6.323, 6.223, 6.1399, 6.06786, 6.007,  5.9626, 5.929]
#  TODO: artifacts for Oct 01, find last descending pixel
width = pd.read_csv(stuff+'width.csv')
length = pd.read_csv(stuff+'length.csv')
ang = pd.read_csv(stuff+'angle.csv')
# parameters and baseline correction
rad = 83
ver = '2'
zoom = False
pkl = np.load(stuff + 'peaks'+str(rad)+'.pkl', allow_pickle=True)
PP = pkl[0]
data = np.load(stuff + 'per_day_smooth'+ver+'.pkl', allow_pickle=True)
for day in range(7):
    layer = data[:, :, day]
    bl = np.nanpercentile(layer, 45)
    layer = layer - bl
    if day == 4:
        layer = layer * 1.36425565 # 1.316422045828965
    data[:, :, day] = layer
## loop
clim1 = 1
center = [302, 307]
# center  = np.unravel_index(np.nanargmax(data[:,:,0]), data.shape[:2])
save = False
plt.figure()
for day in range(7):
    peaks = ridges(data[:, :, day], range(25, 191), tangential_smooth=10, prominence=0.01, height=0.04, center=center,
                       smooth=True, plot=False)
    peakmap = data[:,:,day].copy()
    mx = peakmap.max()
    for peak in peaks:
        if peakmap[peak[0],peak[1]] < 0.5:
            peakmap[peak[0],peak[1]] = 1000
        else:
            peakmap[peak[0], peak[1]] = -1000
    plt.subplot(2,4, day+1)
    img2 = peakmap
    img = np.zeros((img2.shape[0],img2.shape[1], 3))
    img[:, :, 0] = img2
    for rgb in range(3):
        img[:, :, rgb] = img2
    im = img[:, :, 0].copy()
    im[np.abs(im) == 1000] = 1000
    img[:, :, 0] = im
    im = img[:, :, 1].copy()
    im[np.abs(im) == 1000] = -1000
    img[:, :, 1] = im
    im = img[:, :, 2].copy()
    im[np.abs(im) == 1000] = -1000
    img[:, :, 2] = im
    img = img*2
    img[img > clim1] = clim1
    img[img < 0] = 0
    img[np.isnan(img)] = 0
    # if save:
    date = mmddu[day].replace('09', 'Sep ').replace('10', 'Oct ')
    # plt.title(mmddu[day].replace('09','Sep ').replace('10','Oct '))
    # plt.text(50,50,mmddu[day].replace('09','Sep ').replace('10','Oct '),color='white')
    img = putText(img, date, [50, 50], FONT_HERSHEY_DUPLEX, 1, color=[1, 1, 1])
    # plt.figure()
    plt.imshow(img)
    plt.clim(0, clim1)
    plt.axis('off')
    if save:
        plt.imsave(stuff + '0' + str(day) + '.png', img)
    del img
    plt.plot(center[1], center[0], 'or', markersize=3)
    if day > 0:
        for il in range(3):
            w = width[width.columns[il+1]][day-1]/pix_size[day]
            xy0 = np.flipud(PP[day][il])  - center
            x0 = ((w/2)**2 + rad**2)**0.5
            ang0 = np.arctan(xy0[1]/xy0[0])  # angle from center to peak
            xx1 = np.cos(np.deg2rad(90)+ang0) * w/2
            yy1 = np.sin(np.deg2rad(90)+ang0) * w/2
            xx2 = np.cos(np.deg2rad(90)-ang0) * w/2
            yy2 = -np.sin(np.deg2rad(90)+ang0) * w/2
            # plt.plot(PP[day][il][1],PP[day][il][0],'og')
            plt.plot([xx1 + PP[day][il][1], xx2 + PP[day][il][1]], [yy1 + PP[day][il][0], yy2 + PP[day][il][0]], 'b')
            a = ang[ang.columns[il+1]].loc[day-1]
            l = length[length.columns[il+1]].loc[day-1]
            yl = np.cos(np.deg2rad(a)) * l / pix_size[day]
            xl = np.sin(np.deg2rad(a)) * l / pix_size[day]
            if il > 0 or day ==1:
                plt.plot(center[1] + xl,center[0] - yl,'og',markersize=3)
plt.subplot(2,4, day+2)
white = np.ones((data.shape[0], data.shape[1], 3))
shift = 5
img = putText(white, 'tangential peak', [100, 55], FONT_HERSHEY_DUPLEX, 1, color=[0, 0, 0])
img = putText(img, 'width', [100, 85], FONT_HERSHEY_DUPLEX, 1, color=[0, 0, 0])
img = putText(img, 'length', [100, 115], FONT_HERSHEY_DUPLEX, 1, color=[0, 0, 0])
img = putText(img, 'center', [100, 145], FONT_HERSHEY_DUPLEX, 1, color=[0, 0, 0])
plt.imshow(img, cmap='gray')
plt.plot([0,40],[50,50],'r')
plt.plot([0, 40],[85, 85], 'b', 2)
plt.plot(20, 115, 'og')
plt.plot(20, 145, 'or')
plt.clim(0,1)
plt.axis('off')

## lengths
datetick = [x.replace('09', 'Sep ').replace('10', 'Oct ') for x in mmddu]
xsh = [-0.25, 0, 0.25]
co = [[31 / 255, 119 / 255, 180 / 255], [1, 127 / 255, 14 / 255], 'g']
plt.figure()
for ib in [0, 1, 2]:
    l = length[length.columns[ib+1]].to_numpy()
    if ib == 0:
        l[1:] = 1400
    plt.bar(np.arange(1, 7)+xsh[ib], l, 0.25)
ax = plt.gcf()
ax.axes[0].yaxis.grid()
plt.ylabel('length (km)')
plt.ylim(0, 1400)
# plt.xlabel('night')
plt.title('Tail length by day')
plt.legend(['tail A', 'tail B', 'tail C'])
plt.yticks(np.arange(0,1400, 100))
plt.xticks(np.arange(1,7), datetick[1:])
##
w = pkl[3]
threshold = np.array([0.05, 0.1, 0.2, 0.3])

fig = plt.figure()
fig.set_size_inches(14, 5)
plt.suptitle('tail widths by observation night and flux threshold (color)')
for tail in range(3):
    plt.subplot(1, 3, tail+1)
    for ith in range(4):
        plt.plot(np.arange(1,7), pix_size[1:]*np.squeeze(w[:, tail, ith])*2)
    plt.ylim(0, 700)
    plt.ylabel('width (km)')
    plt.grid()
    plt.xticks(range(1,7), datetick[1:])
    plt.title('tail '+'ABC'[tail])
    if tail == 1:
        plt.legend(threshold/5)

## rotaion
datetick = [x.replace('09', 'Sep ').replace('10', 'Oct ') for x in mmddu]
xsh = [-0.25, 0, 0.25]
co = [[31 / 255, 119 / 255, 180 / 255], [1, 127 / 255, 14 / 255], 'g']
plt.figure()
for ib in [0, 1, 2]:
    rot = np.diff(ang[ang.columns[ib+1]].to_numpy())
    rot[rot > 180] = 360-rot[rot > 180]
    if ib == 2:
        rot[0] = 3.3
    plt.bar(np.arange(2, 7)+xsh[ib], rot, 0.25)
ax = plt.gcf()
ax.axes[0].yaxis.grid()
plt.ylabel('rotation (deg)')
plt.ylim(-4.5, 4.5)
# plt.xlabel('night')
plt.title('Tail rotation from previous night')
plt.legend(['tail A', 'tail B', 'tail C'])
plt.yticks(np.arange(-4,5))
plt.xticks(np.arange(2,7), datetick[2:])
plt.xlabel('rotation end date')
##
plt.figure()
for ib in [0, 1, 2]:
    rot = np.diff(ang[ang.columns[ib+1]].to_numpy())
    rot[rot > 180] = 360-rot[rot > 180]
    if ib == 2:
        rot[0] = 3.7
    rot = [0]+list(np.cumsum(rot))
    plt.plot(np.arange(1, 7), rot)
# ax = plt.gcf()
# ax.axes[0].yaxis.grid()
plt.grid()
plt.ylabel('rotation (deg)')
plt.ylim(-2, 13)
# plt.xlabel('night')
plt.title('Cumulative tail rotation')
plt.legend(['tail A', 'tail B', 'tail C'])
plt.yticks(np.arange(-2,13))
plt.xticks(np.arange(1,7), datetick[1:])
# plt.xlabel('rotation end date')