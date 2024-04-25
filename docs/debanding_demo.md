## Explaining astro_deband function for removing banding (1/f noise) from JWST data
Download the example data of sunburst arc, file jw02555-o003_t009_nircam_clear-f115w_i2d.fits, from [MAST](https://mast.stsci.edu/portal/Download/file/JWST/product/jw02555-o003_t009_nircam_clear-f115w_i2d.fits) or via [jwstfeed](https://jwstfeed.com/StsciImage/ViewImage?fullImagePath=https://mast.stsci.edu/portal/Download/file/JWST/product/jw02555-o003_t009_nircam_clear-f115w_i2d.jpg)<br>
### Banding noise
Banding noise is a pattern of thin horizontal lines visible in JWST NIRCam data, frequently for narrowband filters.
```python

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
from astropy.io import fits
import numpy as np

hdu0 = fits.open('jw02555-o003_t009_nircam_clear-f115w_i2d.fits')
data = hdu0[1].data[3400:4400, 2900:3900]
hdu0.close()
data[data > 1] = 1
plt.imsave('raw.png', data, origin='lower', cmap='gray')
```
![raw image](deband/raw.png)
### G'Mic
A tool sometimes used to clean the artifact, is [G'MIC](https://gmic.eu/).
It can be used by a GUi but there is also a command line implementation. There is also a python version but it did not work for me. 
```python
import os
command = 'gmic raw.png -banding_denoise_v2 "10","0","10","20","1","0","0","50,50" -o gmic.png'
os.system(command)
```
A few issues:
* G'MIC doesn't care for direction, and cleans also vertical lines.
* There are dark pixels left and right of some stars, a result of overshoot, cleaning too much by removing real signal mistaken to be noise.
* gmic operates on an image file such as png, and not on the original .fits file data. Normally image files are created after some color stretching. Here I only removed everything above 1 (see `data[data > 1] = 1` above), but we want to keep the data for stretching after we clean it.
The process is fast, took 2.7 sec for this 1000 x 1000 pix square. One may want to play with the parameters for better cleaning. the "1" argument refers to the first choice of tile size, 64pix. You can try other sizes. 
![raw image](deband/gmic.png)
# deband_layer, median subtraction
```python
from astro_deband import deband_layer
# read the data from fits, skip the data[data > 1] = 1 line
clean = deband_layer(data)
plt.imsave('median.png', clean, origin='lower', cmap='gray')
```
I wrote deband layer with the aim of removing the median of a selected window ( lets use the default length of 101 pixels) row by row. So for every point int he image, the median is computed using 50 pixels left and right. However, we want to keep the values if they are not particulat to the horizontal direction. I do this by creating a smoothed image combining the median filter for both orientations. In other words, I remove the fine lines while keeping the lower frequencies not specific to the horizontal direction.
* The process too 4.2sec, slower than gmic
* the overshooting is worse, large dark patches when two bright sources are horizontally aligned.
* The edges were lost, half a window for each direction (50pix)
![deband layer using median subtraction](deband/median.png)
# percentile
Subtracting the 10th percentile instead of the median improves the overshoot issue.
```python
clean = deband_layer(data, func=np.percentile, prct=10)
plt.imsave('percentile.png', clean, origin='lower', cmap='gray')
```
* overshoot (dark patches between stars) is better but not completely solved.
* we still lose half a window on all sides
* cleaning took longer, 5.6sec
* There is still a problem with the arc at the bottom of the image. The signal itself is a narrow horizontal line, and is subtracted from the image.
![deband layer using percentile instead of median](deband/percentile.png)
# nanpercentile
When the data includes nans nanpercentile must be used to avoid 101px size holes around every nan pixel. However it is also helpful for not losing the half-window strips at the margins.
```python
clean = deband_layer(data, func=np.nanpercentile, prct=10)
plt.imsave('nanpercentile.png', clean, origin='lower', cmap='gray')
```
* We still have distorted arc
* The process was 34 times slower. it took 190.1sec, big differance.
![deband layer using nanpercentile to keep the margins](deband/nanpercentile.png)
# Larger window size
To solve the distortion of the arc, we need a window long enough to include 10% baseline, dark sky. Here I used `win=401`. 
```python
clean = deband_layer(data, func=np.nanpercentile, prct=10, win=401)
plt.imsave('nanpercentile_win=401.png', clean, origin='lower', cmap='gray')
```
The arc is preserved, and areas between stars have negligible darkening, if any.
* The window might be too long to adapt for changes in noise levels along the horizontal lines 
* it took 179.7sec.
![deband layer using nanpercentile to keep the margins](deband/nanpercentile_win=401.png)
# Conclusion
`deband_layer` lets me clean the data from banding noise using the raw data, before color stretch. Here I used a challanging image in order to test edge cases, but often using the quicker median is sufficient.<br>
Further improvements could be made. I plan to use nan approach to treat the edges onlty, thus making the process fast while keeping the edges intact. I also want to try removing medians, or percentile 10, from variable window length. This makes sense because the images have rectangular noise patterns that may be cleaned hole from side to side.
# The code
see [astro_deband.py](../astro_deband.py) or here:
```python
import numpy as np
# from astropy.io import fits


def smooth_width(layer, win=101, prct=50, func=np.median):
    '''
    smooth image from left to right, uses nanmedian
    Args:
        func: np function for smoothing
            median, nanmedian, percentile or nanpercentile. nan is slower but without it you loose 50 pixels on all sides
        prct: int
            when removing percentile specifies q. recommended prct=10
        layer: 2D ndarray
        win: int

    Returns:
        smoothed data
    '''
    if func == np.median or func == np.nanmedian:
        args = {'axis': 1}
    else:
        args = {'q': prct, 'axis': 1}
    half0 = int(win / 2)
    half1 = win - half0
    smoothed = layer.copy()
    for ii in range(smoothed.shape[0]):
        toavg = np.nan * np.ones((layer.shape[1] + win - 1, win))
        for shift in np.arange(win):
            toavg[shift:layer.shape[1] + shift, shift] = layer[ii, :]
        smoothed[ii, :] = func(toavg, **args)[half0:-half1 + 1]
        # smoothed[ii, :] = np.nanpercentile(toavg, prct, axis=1)[half0:-half1 + 1]
        print(f'{ii}/{smoothed.shape[0]-1}', end='\r')
    return smoothed


def deband_layer(layer, win=101, prct=10, func=np.median):
    lp = smooth_width(layer, win=win, prct=prct, func=func)
    hp = layer - lp
    lp = smooth_width(lp.T, win=win, prct=prct, func=func).T
    clean = lp + hp
    clean[clean < 0] = 0
    return clean

```