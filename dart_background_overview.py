import matplotlib.pyplot as plt
import numpy as np
# %matplotlib qt
from astro_utils import *
from astro_fill_holes import *
import os
import pandas as pd
# import cv2
# %matplotlib qt

os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/')
stuff = '/home/innereye/astro/dart/'
path = np.asarray(list_files('/home/innereye/Dropbox/Moris_20220926-20221002/', '*.fits'))
kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data
mmdd = np.asarray([x[9:13] for x in path])
mmddu = np.unique(mmdd)
## get center of didymos for al images
halfdid = 150
data = np.zeros((halfdid*2, halfdid*2, 7))

for day in range(7):
    date = mmddu[day]
    filt = []
    pathc = path[mmdd == date]
    filename = stuff + 'xy'+date+'.csv'
    dfxy = pd.read_csv(filename)
    if day == 0:
        df = dfxy
    else:
        df = df.append(dfxy, ignore_index=True)


peak = []
filt = []
sky = []
airmass = []
for row in range(len(df)):
    hdu = fits.open(df['file'].loc[row])
    img = hdu[0].data.copy()
    sky.append(np.median([np.mean(img[0:16, 30:46]),
                          np.mean(img[0:16, -16:]),
                          np.mean(img[-46:-30, 30:46]),
                          np.mean(img[-46:-30, -16:])]))
    if df['x'].loc[row] > 0:
        peak.append(img[df['x'].loc[row],df['y'].loc[row]])
    else:
        peak.append(0)
    filt.append(hdu[0].header['FILTER'])
    airmass.append(hdu[0].header['TCS_AM'])
    print(row)


df['peak'] = peak
df['sky'] = sky
df['airmass'] = airmass
df['filter'] = filt
df.to_csv(stuff+'meta.csv', index=False)

plt.figure()
plt.subplot(1,2,1)
plt.plot(df['airmass'][df['filter'] == 'Johnson_V'],df['sky'][df['filter'] == 'Johnson_V'],'.', label='sky')
plt.xlabel('Air Mass')
plt.ylabel('light')
plt.title('Sky against Airmass for Johnson_V')
plt.subplot(1,2,2)
plt.plot(df['airmass'][df['filter'] == 'Johnson_V'],df['peak'][df['filter'] == 'Johnson_V'],'.', label='peak')
plt.title('Peak against Airmass for Johnson_V')
# plt.legend()
plt.xlabel('Air Mass')
plt.ylabel('light')

plt.figure()
for ii, day in enumerate(mmddu):
    idx = (df['filter'] == 'Johnson_V') & (df['x'] > 0) & df['file'].str.contains(day)
    plt.plot(list(df['peak'][idx]), '.', label=day)
    plt.xlabel('image number')
    plt.ylabel('peak')
plt.legend()

weather.columns = ['HSTYear', 'HSTMonth', 'HSTDay', 'HSTHour', 'HSTMin', 'HSTSec', 'Temp(C)', 'Dew_Pt(C)', 'RH(%)', 'Wspd(mph)', 'Wdir(deg)', 'Peak_Wspd(mph)', 'Pres(mb)', 'Rain(mm)', 'PW(mm)']