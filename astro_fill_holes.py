from astropy.io import fits
import os
import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
from astropy.convolution import Ring2DKernel, Gaussian2DKernel, convolve
from scipy.signal import find_peaks
from skimage.morphology import disk
# from scipy.ndimage import median_filter
from tqdm import tqdm

root = os.environ['HOME']+'/astro/'

def hole_xy(layer, x_stddev=4):
    """
    find xy for holes by filtering a mask = layer <= 0, then finding peaks in the filtered mask.
    Parameters
    ----------
    layer: 2D np.ndarray
    x_stddev: int, passed as property of Gaussian2DKernel and sets the size. kerned size is 8 * x_stddev + 1

    Returns
    -------
    peaks: N by 2 np.ndarray, with x and y of hole position
    """
    # Find x, y for holes in a filtered mask of layer == 0
    # FIXME: consider to points or more with the same max value
    kernel = Gaussian2DKernel(x_stddev=x_stddev)
    # kerned size is 8*std+1
    layer[layer < 0] = 0
    zeros = layer == 0
    if np.sum(zeros) <= 0:
        print('no holes?')
        return []
    zeros = layer == 0
    zeros_smoothed = convolve(zeros, kernel=kernel.array)
    hor = np.zeros(layer.shape, bool)
    for ii in range(layer.shape[0]):
        peaks = find_peaks(zeros_smoothed[ii, :])[0]
        if len(peaks) > 0:
            peaks = peaks[zeros_smoothed[ii, peaks] < 1]
        # if len(peaks) > 0:
        hor[ii, peaks] = True
    ver = np.zeros(layer.shape, bool)
    for jj in range(layer.shape[1]):
        peaks = find_peaks(zeros_smoothed[:, jj])[0]
        if len(peaks) > 0:
            peaks = peaks[zeros_smoothed[peaks, jj] < 1]
        # if len(peaks) > 0:
        ver[peaks, jj] = True
    peaks = np.where(ver * hor)
    peaks = np.asarray(peaks).T
    return peaks

def find_peaks2d(img, x_stddev=4):
    kernel = Gaussian2DKernel(x_stddev=x_stddev)
    print('smoothing')
    smoothed = convolve(img, kernel=kernel.array)
    print('done')
    hor = np.zeros(img.shape, bool)
    for ii in range(img.shape[0]):
        peaks = find_peaks(smoothed[ii, :])[0]
        if len(peaks) > 0:
            peaks = peaks[smoothed[ii, peaks] < 1]
        # if len(peaks) > 0:
        hor[ii, peaks] = True
    ver = np.zeros(img.shape, bool)
    for jj in range(img.shape[1]):
        peaks = find_peaks(smoothed[:, jj])[0]
        if len(peaks) > 0:
            peaks = peaks[smoothed[peaks, jj] < 1]
        # if len(peaks) > 0:
        ver[peaks, jj] = True
    peaks = np.where(ver * hor)
    peaks = np.asarray(peaks).T
    return peaks



def hole_size(layer, xy, plot=False):
    """
    Here we look for hole size by climbing out of a "crater" to 4 directions, to the rim of the crater.
    Thus we get 4 estimates of hole size.
    Parameters
    ----------
    layer: the image as a 2D np.ndarray
    xy: location of holes, N by 2 np.ndarray
    plot: bool, True if you want to see rim detection

    Returns
    -------
    rad_rim: N by 4 np.ndarray, radius from hole center to rim. 4 radii for left right up and down.
    """
    rad_rim = np.zeros((xy.shape[0], 4))
    rad_rim[...] = np.nan
    # rad_zeros = np.zeros(xy.shape[0])
    for hole in range(xy.shape[0]):
        # Look for the first sample after the peak before and after hole center for y and x
        # below
        where = np.where(layer[xy[hole, 0], 1:xy[hole, 1]] - layer[xy[hole, 0], :xy[hole, 1]-1] > 0)[0]
        if len(where) > 0:  # edge issues, maybe zeros
            y0 = where[-1] + 1
            rad_rim[hole, 0] = xy[hole, 1] - y0
        # above
        where = np.where(layer[xy[hole, 0], xy[hole, 1]+1:] - layer[xy[hole, 0], xy[hole, 1]:-1] > 0)[0]
        if len(where) > 0:
            y1 = where[0] + xy[hole, 1] + 1
            rad_rim[hole, 1] = y1 - xy[hole, 1]
        # left
        where = np.where(layer[1:xy[hole, 0], xy[hole, 1]] - layer[:xy[hole, 0] - 1, xy[hole, 1]] > 0)[0]
        if len(where) > 0:  # edge issues, maybe zeros
            x0 = where[-1] + 1
            rad_rim[hole, 2] = xy[hole, 0] - x0
        # right
        where = np.where(layer[xy[hole, 0]+1:, xy[hole, 1]] - layer[xy[hole, 0]:-1, xy[hole, 1]] > 0)[0]
        if len(where) > 0:
            x1 = where[0] + xy[hole, 0] + 1
            rad_rim[hole, 3] = x1 - xy[hole, 0]
        # if np.any(rad_rim < 0):
        #     print('dbg neg idx')
        # if hole == 47:
        #     print('big one')
    rad_rim[rad_rim[:, 0] + xy[:, 0] > layer.shape[0], 0]
    if plot:
        tmp = layer.copy()
        mx = tmp.max()
        for ii in range(xy.shape[0]):
            tmp[xy[ii, 0],xy[ii, 1]] = mx
            for jj in range(4):
                if not np.isnan(rad_rim[ii,jj]):
                    if jj == 0:
                        tmp[xy[ii, 0], xy[ii, 1] - int(rad_rim[ii, jj])] = mx
                    elif jj == 1:
                        tmp[xy[ii, 0], xy[ii, 1] + int(rad_rim[ii, jj])] = mx
                    elif jj == 2:
                        tmp[xy[ii, 0] - int(rad_rim[ii, jj]), xy[ii, 1]] = mx
                    else:
                        tmp[xy[ii, 0] + int(rad_rim[ii, jj]), xy[ii, 1]] = mx
        plt.figure()
        plt.imshow(tmp, origin='lower', cmap='gray')
        plt.axis('off')
        plt.show(block=False)
    return rad_rim

def hole_disk_fill(img, xy, size, larger_than=2, allowed=1/3):
    """
    fill holes with a disk
    check if there are at least 3 similar radii (consistent circular size),
    then replace zeros area with a disk. the value of the disk is local maximum
    Parameters
    ----------
    img: 2D np.ndarray
    xy: N by 2 np.ndarray
    size: N by 4 np.ndarray
    larger_than: lower limit for radius size, dont try to fix small holes
    allowed: when evaluating number of valid radii per hole, allow "allowed" variability

    Returns
    -------
    filled: the fixed image
    """
    # fill holes larger than larger_than with a circle according to xy and size
    # TODO: realign center from xy to middle of rim points (xy +- size)
    # TODO: remove leftover zeros with dark neighbors
    filled = img.copy()
    # allowed = 1/3  # how much variability in size is allowed
    # size1 = np.zeros(xy.shape[0])
    for ii in range(xy.shape[0]):
        sz = size[ii, ~np.isnan(size[ii, :])]
        if len(sz) > 2:
            sz = sz[np.abs(sz/np.median(sz)-1) <= allowed]
            if len(sz) > 2:
                sz = int(np.ceil(np.mean(sz)))
                in_frame = (xy[ii,0] - sz > 0) and (xy[ii, 1] - sz > 0) and (xy[ii, 0] + sz < img.shape[0]) \
                           and (xy[ii, 1] + sz < img.shape[1])
                if sz > larger_than and in_frame:
                    mask = disk(sz + 1, bool)
                    mask = mask[1:-1, 1:-1]
                    fill = img[xy[ii, 0]-sz:xy[ii, 0]+sz+1, xy[ii, 1]-sz:xy[ii, 1]+sz+1]
                    fill[mask] = fill.max()
                    filled[xy[ii, 0]-sz:xy[ii, 0]+sz+1, xy[ii, 1]-sz:xy[ii, 1]+sz+1] = fill
    return filled


def hole_conv_fill(img, n_pixels_around=4, ringsize=15, clean_below_local=0.75, clean_below=1):
    """
    fill (small) holes with local mean. local mean is computed after ignoring zeros.
    zero and negative values are replaced. neighbors are also replaced if smaller than 75% of local mean.
    designed to fill dead pixels, not stars
    Parameters
    ----------
    img: the input 2D image
    n_pixels_around: int, how far should neighbors be from zeros
    x_stddev: int, passed as property of Gaussian2DKernel and sets the size. kerned size is 8 * x_stddev + 1.

    Returns
    -------
    img: the fixed image
    """
    # kernel = Gaussian2DKernel(x_stddev=x_stddev)
    kernel = Ring2DKernel(ringsize, 3)
    # conv = median_filter(img, footprint=kernel.array)
    img[np.isnan(img)] = 0  # turn nans to zeros for later filling
    zer = np.where(img <= 0)
    zer = np.asarray(zer).T
    img[img == 0] = np.nan  # turn zeros to nans to ignore when computing fill values
    conv = convolve(img, kernel)
    med = np.nanmedian(img)
    img[np.isnan(img)] = 0  # change back to zeros to allow operands

    if n_pixels_around is None or n_pixels_around == 0:
        img[img <= clean_below*med] = conv[img <= clean_below*med]
    else:
        idx = list(range(-n_pixels_around, n_pixels_around+1))
        for ii in range(zer.shape[0]):
            img[zer[ii, 0], zer[ii, 1]] = conv[zer[ii, 0], zer[ii, 1]]
            for jj in idx:
                x = zer[ii, 0] + jj
                for kk in idx:
                    y = zer[ii, 1] + kk
                    if x == 511 and y == 88:
                        a=1  # debug stop
                    if x > -1 and y > -1 and x < img.shape[0] and y < img.shape[1]:
                        if (img[x, y] < conv[x, y] * clean_below_local) and img[x, y] < med * clean_below:
                            img[x, y] = conv[x, y]
    img[np.isnan(img)] = 0
    return img



if __name__ == '__main__':
    path = root+'jw01288-o001_t011_nircam_clear-f480m_cropped.fits'
    # hdu = fits.open(path[8])
    hdu = fits.open(path)
    # img = hdu[1].data[3800:5000, 5600:7000]
    img = hdu[0].data
    xy = hole_xy(img)
    size = hole_size(img, xy, plot=False)
    orig = img.copy()
    filled = hole_disk_fill(img, xy, size, larger_than=3)
    # filled = hole_conv_fill(filled, x_stddev=4)
    filled = hole_conv_fill(filled, n_pixels_around=3, ringsize=15, clean_below_local=0.75, clean_below=2)
    plt.figure();
    plt.imshow(filled, origin='lower');
    plt.clim(0, 1000);
    plt.show(block=False)
    conved = orig.copy()
    conved = hole_conv_fill(conved)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(orig[0:400, 0:400], origin='lower', cmap='gray')
    plt.axis('off')
    plt.title('orig')
    plt.subplot(1, 3, 2)
    plt.imshow(filled[0:400, 0:400], origin='lower', cmap='gray')
    plt.axis('off')
    plt.title('disks + conv')
    plt.subplot(1, 3, 3)
    plt.imshow(conved[0:400, 0:400], origin='lower', cmap='gray')
    plt.axis('off')
    plt.title('conv')
    plt.show(block=False)
    print('tada')


