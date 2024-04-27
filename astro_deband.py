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



if __name__  == '__main__':
    print('see deband_demo.py')