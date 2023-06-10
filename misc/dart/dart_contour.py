import matplotlib.pyplot as plt
import numpy as np
%matplotlib qt
from astro_utils import *
from astro_fill_holes import *
import os
import pandas as pd
# import pickle
## Take the first day data (before impact) to compute flat field
os.chdir('/home/innereye/Dropbox/Moris_20220926-20221002/')
stuff = '/home/innereye/astro/dart/'
path = list_files('/home/innereye/Dropbox/Moris_20220926-20221002/', '*.fits')
kernel = Gaussian2DKernel(x_stddev=3)  # a gaussian for smoothing the data
mmdd = np.asarray([x[9:13] for x in path])
mmddu = np.unique(mmdd)
data = np.load(stuff+'per_day_raw.pkl', allow_pickle=True)
levs = 10.0**np.arange(-2,2.6,0.5)
plt.figure()
for ii in range(7):
    d = data[:,:,ii].copy()
    bl = (np.median(d[15:50,15:50])+np.median(d[-50:-15,-50:-15]))/2
    d = d - bl
    if ii == 4:
        d = d*1.5
    d[d < 0.01] = 0.01
    plt.subplot(2,4,ii+1)
    cs = plt.contourf(d, levs, norm = matplotlib.colors.LogNorm(), origin='lower')
    plt.axis('off')
    plt.axis('square')
plt.subplot(2,4,8)
plt.axis('off')
plt.colorbar(format='%.2f')


## plot daily midline
jet = matplotlib.cm.get_cmap('jet', 9)
jet = jet(np.linspace(0, 1, 9))
plt.figure()
for ii in range(7):
    plt.plot(data[150,:,ii]-np.mean(data[150,22:30,ii]), label=mmddu[ii], color=jet[ii+1,:3])
plt.xlim(120, 180)
plt.ylim(0, 300)
plt.grid()
plt.legend()
