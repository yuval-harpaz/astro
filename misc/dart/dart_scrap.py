import matplotlib.pyplot as plt
import numpy as np
# %matplotlib qt
from astro_utils import *
from astro_fill_holes import *
import os
import pandas as pd
# import cv2
dist = np.asarray([11.41, 11.23, 11.08, 10.95, 10.84, 10.76, 10.70])*10**6
pix_size = np.tan(np.deg2rad(0.1143/2/3600))*dist*2

os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/')
stuff = '/home/innereye/astro/dart/'
path = np.asarray(list_files('/home/innereye/Dropbox/Moris_20220926-20221002/', '*.fits'))
kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data
mmdd = np.asarray([x[9:13] for x in path])
mmddu = np.unique(mmdd)
## get center of didymos for al images

day = 5
date = mmddu[day]
filt = []
pathc = path[mmdd == date]
filename = stuff + 'xy'+date+'.csv'
dfxy = pd.read_csv(filename)
# for ii in range(len(pathc)):
#     f = pathc[ii]
#     hdu = fits.open(f)
#     filt.append(hdu[0].header['FILTER'])
# filt = np.asarray(filt)
# filtu = np.unique(filt)
# print(date)
# for ii in range(len(filtu)):
#     print(filtu[ii]+': '+str(np.sum(filt == filtu[ii])))


## clean V data
# hot = pd.read_csv(stuff+'hot_pixels.csv')
# halfdid = 150
# didV = np.zeros((halfdid*2, halfdid*2, 7))
#
# filt = []
# pathc = path[mmdd == date]
# filename = stuff + 'xy'+date+'.csv'
# dfxy = pd.read_csv(filename)
# xy = np.zeros((len(dfxy),2), int)
# xy[:,0] = dfxy['x'].to_numpy()
# xy[:, 1] = dfxy['y'].to_numpy()
# n = 0
# did_clean = []
df = pd.read_csv(stuff+'meta.csv')
# df = df[df['x'] > 0]
mmdd = np.asarray([x[9:13] for x in df['file']])
filt = np.asarray([x[-1].lower() for x in df['filter']])
# idx = np.where((filt == 'v') & (mmdd == '1001') & (df['x'] > 0) & (df['y'] < 370))[0]
df = df[(filt == 'v') & (mmdd == '1001') & (df['x'] > 0) & (df['y'] < 370)]
file = df['file'].to_numpy()
c = 0
plt.figure()
for ii in range(283):
    c += 1
    if c == 37:
        c = 1
        plt.figure()
    hdu = fits.open(file[ii])
    plt.subplot(6, 6, c)
    plt.imshow(hdu[0].data, origin='lower')
    plt.clim(200, 205)
