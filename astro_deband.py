import numpy as np
# from astropy.io import fits


def smooth_width(layer, win=101, prct=50, func=np.median):
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
        print(f'{ii}/{smoothed.shape[0]-1}', end='\r')
    return smoothed


def deband_layer(layer, win=101, prct=10, func=np.median, flip=False):
    """
    remove 1/f, banding noise. thin stripes.
    func: percentile is safer than median, with nan* you don't lose the edges 
          but it takes ages.

    Parameters
    ----------
    layer : ndarray
        2D image before reproject.
    win : int, optional
        window length for computing noise. The default is 101.
    prct : int, optional
        lower than 50 for precentile or nanpercentile func. The default is 10.
    func : np function, optional
        should be median, anamedian, percentile or nanpercentile. The default is np.median.
    flip : bool, optional
        use True for MIRI images. The default is False.

    Returns
    -------
    clean : ndarray
        data cleaned of stripes.

    """
    if flip:
        layer = layer.T
    lp = smooth_width(layer, win=win, prct=prct, func=func)
    hp = layer - lp
    lp = smooth_width(lp.T, win=win, prct=prct, func=func).T
    clean = lp + hp
    clean[clean < 0] = 0
    if flip:
        clean = clean.T
    return clean



if __name__  == '__main__':
    print('see deband_demo.py')
