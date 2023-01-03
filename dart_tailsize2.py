# %matplotlib qt
import numpy as np
from matplotlib import pyplot as plt
# from astro_utils import movmean
from dart_tools import ridges, sorty
# from astro_fill_holes import *
import os
from cv2 import circle, line
from sympy import symbols, Eq, solve


os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/')
stuff = '/home/innereye/astro/dart/'

## parameters and baseline correction
ver = '2'
gray = False
save = False
smooth = True  # False for raw, True for loading conv data, 2 for 9 pix square smoothing
zoom = False
clim1 = 1
height = 0.01
tang_smooth = 10
if smooth == True:
    data = np.load(stuff + 'per_day_smooth'+ver+'.pkl', allow_pickle=True)
    prom = 0.01
else:
    data = np.load(stuff + 'per_day_raw'+ver+'.pkl', allow_pickle=True)
    prom = 0.01
    height = 0.1


##  check light per day to correct Sep 30
size = np.zeros((7,18))
ratio = np.zeros((7,18))
for day in range(7):
    layer = data[:, :, day]
    for thr in range(2,20):
        size[day,thr-2] = np.sum(layer > thr)
        ratio[day,thr-2] = size[day,thr-2]/size[0,thr-2]

m = np.median((ratio[3, :] + ratio[5, :])/2/ratio[4, :])
# m = 1.4285230221120706
##  read data and subtract percentile
bl = np.zeros(7)
for day in range(7):
    layer = data[:, :, day]
    bl[day] = np.nanpercentile(layer, 45)
    layer = layer - bl[day]
    if day == 4:
        layer = layer * m
    if smooth == 2:  # smooth a little anyway
        for jj in range(1,298):
            for kk in range(1, 298):
                layer[jj,kk] = np.nanmean(layer[jj-1:jj+2,kk-1:kk+2] )
    data[:, :, day] = layer


## loop
LL = []
PP = []
XY = []
center = [302, 307]
rad = 70
# plt.figure()
for day in range(7):
    peaks = ridges(data[:, :, day], rad, tangential_smooth=10, prominence=prom, height=height, center=center, smooth=True, plot=False)
    ll = []
    means = []
    xy = []
    for pk in peaks:
        a = (pk[1]-center[1])/(pk[0]-center[0])
        b = pk[1]-pk[0]*a
        a_ = -1/a
        b_ = pk[1]-pk[0]*a_
        x, y = symbols('x y')
        eq1 = Eq(((x-pk[0])**2+(y-pk[1])**2)**0.5, 50)
        eq2 = Eq(x*a_+b_, y)
        sol = solve((eq1, eq2), (x, y))
        sol[0] = [float(sol[0][0]),float(sol[0][1])]
        sol[1] = [float(sol[1][0]),float(sol[1][1])]
        imgl = line(np.zeros((data.shape[0],data.shape[1],3)),
                    np.flipud(np.round(sol[0]).astype(int)),
                    np.flipud(np.round(sol[1]).astype(int)),
                    (1,0,0),
                    1)
        # blue = blue + imgl.copy()
        x, y = np.where(imgl[:,:,0])
        y = sorty(x, y, 1)
        xy.append(np.zeros((len(x),2)))
        xy[-1][:, 0] = x
        xy[-1][:, 1] = y
        ll.append(data[x, y, day])
        means.append(np.mean(ll[-1][20:-20]))
    order = np.argsort(-np.array(means))
    ll = [ll[order[mm]] for mm in range(len(ll))]
    peaks = [peaks[order[mm]] for mm in range(len(peaks))]
    xy = [xy[order[mm]] for mm in range(len(xy))]
    # blue = blue[:, :, 0]
    # img[:,:,2] = img[:,:,2]+blue
    # plt.imshow(img);plt.clim(0,1)
    dash = np.median(data[:,:,day])
    # plt.subplot(2, 4, day+1)
    # for il in range(3):
    #     l = ll[il]
    #     xx = np.linspace(-50, 50, len(l))
    #     plt.plot(xx,l)
    #     plt.plot([xx[0],xx[-1]], [dash*10,dash*10],'k:')
    #     plt.plot([xx[0], xx[-1]], [5*dash, 5*dash], 'k:')
    # plt.ylim(0, 1)
    # plt.title('night '+str(day))
    A = np.argmax([peaks[0][1], peaks[1][1], peaks[2][1]])
    B = np.argmax([peaks[0][0], peaks[1][0], peaks[2][0]])
    C = np.argmin([peaks[0][0], peaks[1][0], peaks[2][0]])
    if day > 0:
        order = [A, B, C]
    else:
        order = [0, 1, 2]
    tmp = np.asarray(ll[:3])[order]
    LL.append(tmp)
    tmp = np.asarray(peaks[:3])[order]
    PP.append(tmp)
    tmp = np.asarray(xy[:3])[order]
    XY.append(tmp)

##
plt.figure()
for day in range(7):
    plt.subplot(2,4,day+1)
    img = np.zeros((data.shape[0],data.shape[1],3))
    for layer in range(3):
        img[:,:,layer] = data[:,:,day].copy()
    for ip in range(3):
        img[PP[day][ip][0],PP[day][ip][1],0] = 1
        img[PP[day][ip][0],PP[day][ip][1],1:] = 0
    plt.imshow(img, origin='lower')
    for ip in range(3):
        plt.text(PP[day][ip][1], PP[day][ip][0], 'ABC'[ip], color='g')
    plt.clim(0, 1)
    plt.axis('off')

##
plt.figure()
for day in range(7):
    # dash = np.median(data[:, :, day])
    ll = LL[day]
    plt.subplot(2, 4, day+1)
    for il in range(3):
        l = ll[il]
        xx = np.linspace(-50, 50, len(l))
        plt.plot(xx,l)
        # plt.plot([xx[0],xx[-1]], [dash*10,dash*10],'k:')
        # plt.plot([xx[0], xx[-1]], [5*dash, 5*dash], 'k:')
    plt.ylim(0, 1)
    plt.title('night '+str(day))
    plt.grid()
    plt.yticks(np.arange(0,1,0.1))
    plt.xticks(np.arange(-50,50,10))
    plt.xlabel('tail width (pixels)')
    plt.ylabel('light')
    if day == 0:
        plt.legend(['A','B','C'])

## compute width at different thresholds
threshold = [0.05, 0.1, 0.2, 0.3]
width = np.zeros((6, 3, len(threshold)))
for day in range(1,7):
    ll = LL[day]
    for il in range(3):
        l = ll[il]
        xx = np.linspace(-50, 50, len(l))
        for it, thr in enumerate(threshold):
            left = np.where(l[xx < 0] > thr)[0]
            if len(left) == 0:
                left = 10**6
            else:
                left = -xx[left[0]]
            right = np.where(l[xx > 0] > thr)[0]
            if len(right) == 0:
                right = 10**6
            else:
                right = xx[xx > 0][right[-1]]
            w = np.min([left,right])
            if w == 10**6:
                width[day-1, il, it] = np.nan
            else:
                width[day-1, il, it] = w
# plt.figure();plt.plot(xx,l);plt.plot([-50,50],[thr, thr],'k');plt.plot(-left,thr,'.r');plt.plot(right,thr,'.r')
## plot width
# pix_size =  [6.3 ,6.22 , 6.13, 6.05, 5.97, 5.97, 5.88]

pix_size = [6.323, 6.223, 6.1399, 6.06786, 6.007,  5.9626, 5.929]
plt.figure()
plt.suptitle('tail widths by observation night and luminosity threshold (color)')
for tail in range(3):
    plt.subplot(1, 3, tail+1)
    for ith in range(4):
        plt.plot(np.arange(1,7), pix_size[1:]*np.squeeze(width[:, tail, ith])*2)
    plt.ylim(0, 700)
    plt.ylabel('width (km)')
    plt.grid()
    plt.xticks(np.arange(1,7))
    plt.xlabel('night')
    plt.title('tail '+'ABC'[tail])
    if tail == 1:
        plt.legend(threshold)

##
plt.figure()
plt.suptitle('tail widths by observation night and luminosity threshold (color)')
for ith in range(4):
    plt.subplot(1, 4, ith+1)
    for tail in range(3):
        plt.plot(np.arange(1,7), pix_size[1:]*np.squeeze(width[:, tail, ith])*2)
    plt.ylim(0, 700)
    plt.ylabel('width (km)')
    plt.grid()
    plt.xticks(np.arange(1,7))
    plt.xlabel('night')
    plt.title('threshold '+str(threshold[ith]))
    if ith == 3:
        plt.legend(['tail A','tail B','tail C'])

## tail length
noise_thr = 0.05
ystr = 0.07
length = np.zeros((6, 3))
plt.figure()
for day in range(1,7):
    plt.subplot(2, 4, day+1)
    for il in range(3):
        pk = PP[day][il]
        imgl = line(np.zeros((data.shape[0], data.shape[1], 3)), [(pk[1]-center[1])*10+center[1], (pk[0]-center[0])*10+center[0]],[center[1], center[0]], (1,0,0), 1)
        x, y = np.where(imgl[:, :, 0])
        y = sorty(x, y, 1)
        ll = data[x, y, day]
        # plt.imshow(I, origin='lower')
        x0 = np.max(np.abs(x)-center[0])
        y0 = np.max(np.abs(y)-center[1])
        dist = ((center[0]-x0)**2+(center[1]-y0)**2)**0.5
        xx = np.linspace(dist, 0, len(ll)) * pix_size[day]
        if ll[0] > ll[-1]:
            ll = np.flipud(ll)
        try:
            length[day-1, il] = xx[np.where(ll < noise_thr)[0][-1]]
        except:
            length[day-1, il] = np.max(xx)
        plt.plot(xx, ll)
        plt.ylim(0, 1)
    plt.title('night '+str(day))
    plt.ylabel('light')
    plt.xlabel('length (km)')
    plt.xlim(0, 1750)
    plt.grid()
    plt.plot([0, 2000], [noise_thr, noise_thr], 'k:')
    plt.text(length[day - 1, 1], ystr, str(int(np.round(length[day - 1, 1]))), color=[1, 127 / 255, 14 / 255])
    if day == 1:
        plt.text(length[day - 1, 0], ystr, str(int(np.round(length[day - 1, 0]))), color=[31 / 255, 119 / 255, 180 / 255])
    else:
        plt.text(1550, 0.19, '>1900', color=[31 / 255, 119 / 255, 180 / 255])
    plt.text(length[day - 1, 2], ystr, str(int(np.round(length[day - 1, 2]))), color='g')
plt.suptitle('tail length by observation night')

##
xsh = [-0.25, 0, 0.25]
co = [[31 / 255, 119 / 255, 180 / 255], [1, 127 / 255, 14 / 255], 'g']
plt.figure()
for ib in [0, 1, 2]:
    plt.bar(np.arange(1, 7)+xsh[ib], length[:, ib], 0.25)
ax = plt.gcf()
ax.axes[0].yaxis.grid()
plt.ylabel('length (km)')
plt.ylim(0, 1900)
plt.xlabel('night')
plt.title('length of tails')
plt.legend(['tail A', 'tail B', 'tail C'])
plt.yticks(np.arange(0,1960, 100))

