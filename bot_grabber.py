# https://raw.githubusercontent.com/Rachmanin0xFF/jwst-twitter-bot/main/grabber.py
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt

def to1(x):
    return 0.5 + (2.0 * x - 2.0) / (2 * np.sqrt((2.0 * x - 2.0) * (2.0 * x - 2.0) + 1.0))


def expand_highs(x):
    return np.piecewise(x, [x <= 0.9, x > 0.9],
                        [lambda x: x * 0.8 / 0.9, lambda x: 100.0 / 9.0 * (x - 0.9) ** 2 + 0.8 * x / 0.9])


def image_histogram_equalization(image, number_bins=10000):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = cdf / cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    return image_equalized.reshape(image.shape)


def level_adjust(fits_arr, factor=4.0):
    hist_dat = fits_arr.flatten()
    hist_dat = hist_dat[np.nonzero(hist_dat)]
    zeros = np.abs(np.sign(fits_arr))
    minval = np.quantile(hist_dat, 0.03)
    maxval = np.quantile(hist_dat, 0.98)
    rescaled = (fits_arr - minval) / (maxval - minval)
    rescaled_no_outliers = np.maximum(rescaled, np.quantile(rescaled, 0.002))
    rescaled_no_outliers = np.minimum(rescaled_no_outliers, np.quantile(rescaled_no_outliers, 1.0 - 0.002))
    img_eqd = image_histogram_equalization(rescaled_no_outliers)
    img_eqd = (pow(img_eqd, factor) + pow(img_eqd, factor*2) + pow(img_eqd, factor*4)) / 3.0
    adjusted = expand_highs((img_eqd + to1(rescaled)) * 0.5)
    return np.clip(adjusted * zeros, 0.0, 1.0)


if __name__ == '__main__':
    fname = '/home/innereye/JWST/ngc1512/jw02107-o006_t006_miri_f770w_i2d.fits'
    hdu = fits.open(fname)
    raw = hdu[1].data
    val_arr = level_adjust(hdu[1].data)
    hdu.close()
    plt.figure()
    plt.imshow(val_arr)
