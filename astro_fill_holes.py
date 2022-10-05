from astropy.io import fits
import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.signal import find_peaks
from skimage.morphology import disk

root = os.environ['HOME']+'/astro/'

def hole_xy(layer):
    # Find x, y for holes in a filtered mask of layer == 0
    # FIXME: consider to points or more with the same max value
    kernel = Gaussian2DKernel(x_stddev=4)
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

def hole_size(layer, xy, plot=False):
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
        if np.any(rad_rim < 0):
            print('dbg neg idx')
        if hole == 47:
            print('big one')
    rad_rim[rad_rim[:,0] + xy[:,0]> layer.shape[0], 0]
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

def hole_circle_fill(img, xy, size, larger_than=2):
    # fill holes larger than larger_than with a circle according to xy and size
    # TODO: realign center from xy to middle of rim points (xy +- size)
    # TODO: remove leftover zeros with dark neighbors
    filled = img.copy()
    allowed = 1/3  # how much variability in size is allowed
    # size1 = np.zeros(xy.shape[0])
    for ii in range(xy.shape[0]):
        sz = size[ii, ~np.isnan(size[ii, :])]
        if len(sz) > 2:
            sz = sz[np.abs(sz/np.median(sz)-1) <= allowed]
            if len(sz) > 2:
                sz = int(np.ceil(np.mean(sz)))
                in_frame = (xy[ii,0] - sz > 0) and (xy[ii, 1] - sz > 0) and (xy[ii, 0] + sz <= img.shape[0]) \
                           and (xy[ii, 1] + sz <= img.shape[1])
                if sz > larger_than and in_frame:
                    mask = disk(sz + 1, bool)
                    mask = mask[1:-1, 1:-1]
                    fill = img[xy[ii, 0]-sz:xy[ii, 0]+sz+1, xy[ii, 1]-sz:xy[ii, 1]+sz+1]
                    fill[mask] = fill.max()
                    filled[xy[ii, 0]-sz:xy[ii, 0]+sz+1, xy[ii, 1]-sz:xy[ii, 1]+sz+1] = fill
    return filled


if __name__ == '__main__':
    path = root+'jw01288-o001_t011_nircam_clear-f480m_cropped.fits'
    # hdu = fits.open(path[8])
    hdu = fits.open(path)
    # img = hdu[1].data[3800:5000, 5600:7000]
    img = hdu[0].data
    xy = hole_xy(img)
    size = hole_size(img, xy, plot=False)
    orig = img.copy()
    filled = hole_circle_fill(img, xy, size)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(orig[0:400, 0:400], origin='lower', cmap='gray')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(filled[0:400, 0:400], origin='lower', cmap='gray')
    plt.axis('off')
    plt.show(block=False)
    print('tada')


