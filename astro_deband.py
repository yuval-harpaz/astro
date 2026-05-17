import numpy as np
# from astropy.io import fits


def smooth_width(layer, win=101, prct=50, func=np.median, verbose=False):
    '''
    smooth image from left to right
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
        if verbose:
            print(f'{ii}/{smoothed.shape[0]-1}', end='\r')
    return smoothed


def deband_layer(layer, win=101, prct=10, func=np.median, flip=False, verbose=False):
    """
    Remove 1/f banding noise (thin horizontal or vertical stripes) from a 2D image.

    Algorithm: a row-wise low-pass (stripe estimate) is subtracted to yield a
    high-pass residual; the low-pass is then smoothed column-wise to recover the
    true large-scale background; finally clean = column-smoothed background + residual.
    Negative values produced by the subtraction are clipped to zero.

    Parameters
    ----------
    layer : ndarray
        2D image to clean.
    win : int, optional
        Sliding-window length (in pixels) used by smooth_width to estimate the
        stripe pattern. Larger values capture lower-frequency banding.
        The default is 101.
    prct : int, optional
        Percentile passed to func when func is np.percentile or np.nanpercentile.
        Values below 50 suppress bright point sources from biasing the stripe
        estimate. The default is 10.
    func : callable, optional
        Aggregation function used for smoothing. One of:
        np.median, np.nanmedian, np.percentile, np.nanpercentile.
        Percentile is safer than median near bright sources; nan variants
        preserve edge pixels but are significantly slower.
        The default is np.median.
    flip : bool, optional
        Transpose the image before processing and transpose back afterwards.
        Use True for MIRI images where stripes run vertically instead of
        horizontally. The default is False.
    verbose : bool, optional
        Print row-by-row progress during smoothing. The default is False.

    Returns
    -------
    clean : ndarray
        2D image with stripe noise removed (same shape as layer).
    """
    if flip:
        layer = layer.T
    lp = smooth_width(layer, win=win, prct=prct, func=func, verbose=verbose)
    hp = layer - lp
    lp = smooth_width(lp.T, win=win, prct=prct, func=func, verbose=verbose).T
    clean = lp + hp
    clean[clean < 0] = 0
    if flip:
        clean = clean.T
    return clean



if __name__  == '__main__':
    print('see deband_demo.py')
