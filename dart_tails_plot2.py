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
## parameters and baseline correction
rad = 83
ver = '2'
zoom = False
PP = np.load(stuff + 'peaks'+str(rad)+'.pkl', allow_pickle=True)
data = np.load(stuff + 'per_day_smooth'+ver+'.pkl', allow_pickle=True)
for day in range(7):
    layer = data[:, :, day]
    bl = np.nanpercentile(layer, 45)
    layer = layer - bl
    if day == 4:
        layer = layer * 1.316422045828965
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
                plt.plot(center[1] + xl,center[0] + yl,'og',markersize=3)




