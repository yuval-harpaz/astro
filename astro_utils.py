import pandas as pd
from astropy.io import fits
from astropy import wcs
from reproject import reproject_interp
import os
# import glob
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from astropy.convolution import Ring2DKernel, Gaussian2DKernel, convolve
from scipy.ndimage import median_filter, maximum_filter
from scipy.signal import find_peaks
from astroquery.mast import Observations
# from skimage.morphology import disk
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
# root = __file__[:-14]
# root = list_files.__code__.co_filename[:-14]
root = os.environ['HOME']+'/astro/'

def list_files(parent, search='*_cal.fits', exclude='', include=''):
    """
    List all fits files in subfolders of path.
    """
    os.chdir(root+'data')
    os.chdir(parent)
    path = []
    for found in Path('./').rglob(search):
        path.append(str(found.parent)+'/'+str(found.name))
    if len(include) > 0:
        if type(include) == str:
            include = [include]
        okay = np.zeros(len(path), dtype=bool)
        for ii, p in enumerate(path):
            for i in include:
                if i in p:
                    okay[ii] = True
        path = np.asarray(path)[okay]
    if len(exclude) > 0:
        if type(exclude) == str:
            exclude = [exclude]
        okay = np.ones(len(path), dtype=bool)
        for ii, p in enumerate(path):
            for i in exclude:
                if i in p:
                    okay[ii] = False
        path = np.asarray(path)[okay]
    path = sorted(path)
    return path

def mosaic_xy(path, plot=False):
    '''
    get x y origins of each image
    :param path:
    :return:
    '''
    crval = np.empty((len(path), 3))
    crval[:, :] = np.nan
    dims = np.ndarray((len(path), 1))
    size = []
    pix = []
    for ii, p in enumerate(path):
        hdul = fits.open(p)
        w = wcs.WCS(hdul[1].header)
        if ii == 0:
            w0 = w.copy()
        crval[ii, :len(w.wcs.crval)] = w.wcs.crval
        size.append(w.array_shape)
        pix.append(w.wcs.crpix)
        dims[ii] = len(w.array_shape)
    # if len(np.unique(pix)) > 1:
    #     raise ValueError('Images have different pixel scales')
    if len(np.unique(dims)) > 1:
        raise Exception('Images are not the same dimensions')
    size = np.array(size)
    if np.unique(dims)[0] == 3:
        for isz, sz in enumerate(size):
            if sz[0] < 2:
                sz = sz[1:]
            elif sz[1] == sz[2]:
                sz = sz[1:]
            elif len(sz) == 3:
                if (sz[0] > 500) and (sz[1] < 100) and (sz[2] < 100):
                    sz = sz[1:]
                else:
                    raise Exception('unexpected size '+str(sz))
            else:
                raise Exception('unexpected size '+str(sz))
            size[isz, 0:2] = sz
        size = size[:, 0:2]
    if np.isnan(crval[:,2]).all():
        crval = crval[:, :2]
    # elif np.sum(crval[:, 2]) == 0:
    #     crval = crval[:, :2]
    xy = w0.wcs_world2pix(np.asarray(crval), 1)
    if plot == 2:  # line image locations
        plt.figure()
        for frame, center in enumerate(xy):
            xy_line = np.array([[center[0]-size[frame, 0]/2, center[0]-size[frame, 0]/2, center[0]+size[frame, 0]/2 ,center[0]+size[frame, 0]/2, center[0]-size[frame, 0]/2],
                                [center[1]-size[frame, 1]/2, center[1]+size[frame, 1]/2, center[1]+size[frame, 1]/2, center[1]-size[frame, 1]/2, center[1]-size[frame, 1]/2]]).astype(int).T
            hh = plt.plot(xy_line[:, 0], xy_line[:, 1])
            plt.text(center[0], center[1], str(frame+1), color=hh[0].get_color())
        plt.axis('equal')
        plt.show(block=False)
    return xy, size


def mosaic(data, xy=[], size=[], clip=[], method='overwrite', plot=False):
    if len(data) < 2:
        raise Exception('need at least 2 images')
    if len(xy) == 0:
        if (type(data[0]) == str) | (type(data[0]) == np.str_):
            xy, size = mosaic_xy(data, plot=plot)
    if (type(data[0]) == str) | (type(data[0]) == np.str_):
        path = data
        data = []
        for idat, p in enumerate(path):
            hdul = fits.open(p)
            data.append(hdul[1].data.copy())
    xy = np.asarray(xy)
    # x1 = int(np.max(xy[:, 0])-np.min(xy[:, 0])+xy[:, 0][0]*2+1)
    # y1 = int(np.max(xy[:, 1])-np.min(xy[:, 1])+xy[:, 1][0]*2+1)
    xymin = np.min(xy-size/2,axis=0)
    xy0 = np.zeros(xy.shape)
    xy0[:, 0] = xy[:, 0] - size[:, 0]/2 - xymin[0]
    xy0[:, 1] = xy[:, 1] - size[:, 1]/2 - xymin[1]
    xy0 = np.round(xy0).astype(int)
    xy1 = xy0+np.fliplr(size)

    canvas = np.zeros((np.max(xy1[:, 0]), np.max(xy1[:, 1]))).T
    if method == 'mean':
        exposures = np.zeros(canvas.shape, dtype=int)
    elif method == 'layers' or method == 'median':
        canvas = np.zeros((canvas.shape[0], canvas.shape[1], len(data)))
    if len(clip) == 2:
        mx = clip[1]
        mn = clip[0]
    for idat, d in enumerate(data):
        if len(clip) == 2:
            d[d > mx] = mx
            d[d < mn] = mn
        if method == 'overwrite':  # each image overwrites the previous
            canvas[xy0[idat, 1]:xy0[idat, 1] + size[idat, 0], xy0[idat, 0]:xy0[idat, 0] + size[idat, 1]] = d
        elif method == 'layers' or method == 'median':
            canvas[xy0[idat, 1]:xy0[idat, 1] + size[idat, 0], xy0[idat, 0]:xy0[idat, 0] + size[idat, 1], idat] = d

        elif method == 'mean':  # images are averaged, excluding nan pixels
            nonan = np.zeros(canvas.shape, dtype=bool)
            nn = ~np.isnan(d)
            nonan[xy0[idat, 1]:xy0[idat, 1] + size[idat, 0], xy0[idat, 0]:xy0[idat, 0] + size[idat, 1]] = nn

            exposures[nonan] = exposures[nonan] + 1
            tosum = np.zeros((d.shape[0],d.shape[1],2))
            tosum[:,:,0] = canvas[xy0[idat, 1]:xy0[idat, 1] + size[idat, 0], xy0[idat, 0]:xy0[idat, 0] + size[idat, 1]]
            tosum[:,:,1] = d
            canvas[xy0[idat, 1]:xy0[idat, 1] + size[idat, 0], xy0[idat, 0]:xy0[idat, 0] + size[idat, 1]] = np.nansum(tosum, axis=2)
    if len(clip) == 0:
        mn = np.nanpercentile(canvas, 0.1)
        mx = np.nanpercentile(canvas, 99.9)
    if method == 'mean':
        canvas[np.isnan(canvas)] = 0
        exposures[exposures == 0] = 1
        canvas = canvas / exposures
        canvas[canvas < mn] = mn
    elif method == 'median':
        canvas[canvas == 0] = np.nan
        canvas = np.nanmedian(canvas, axis=2)
        canvas[np.isnan(canvas)] = np.nanmedian(canvas)
    if plot:
        plt.figure()
        if method == 'layers':
            plt.imshow(np.nanmean(canvas, axis=2), cmap='hot', origin='lower')
        else:
            plt.imshow(canvas, cmap='hot', origin='lower')
        plt.axis('equal')
        plt.clim(mn, mx)
        # plt.xlim(0, x1)
        # plt.ylim(0, y1)
        plt.axis('off')
        plt.show(block=False)
    return canvas

def optimize_xy(layers, square_size=100, tests=9, plot=False):
    szx05 = int(layers.shape[0]/2-square_size/2)
    szy05 = int(layers.shape[1]/2-square_size/2)
    square0 = layers[szx05:szx05+square_size, szy05:szy05+square_size, 0]
    # ring = Ring2DKernel(9, 3)
    # square0 = median_filter(square0, footprint=ring.array)
    square0 = median_filter(square0, footprint=np.ones((2, 2)))
    data_vec0 = square0.flatten()
    bestx = [0]
    besty = [0]
    for ii in range(1, layers.shape[2]):
        testx = []
        testy = []
        testr = []
        for jj in range(tests):
            jitter = int(jj-(tests-1)/2)
            for kk in range(tests):
                kitter = int(kk-(tests-1)/2)
                square = layers[jitter + szx05:jitter + szx05 + square_size,
                                kitter + szy05:kitter + szy05 + square_size, ii]
                square = median_filter(square, footprint=np.ones((2, 2)))
                if jj == 2 and kk == 2:
                    plt.figure()
                    plt.imshow(square, cmap='hot')
                    plt.axis('equal')
                    plt.clim(7.45, 10)
                    plt.axis('off')
                    plt.show(block=False)
                data_vec = square.flatten()
                col2 = np.asarray([data_vec0, data_vec]).T
                nans = np.isnan(col2).any(axis=1)
                col2 = col2[~nans,:]
                rr = np.corrcoef(col2[:,0],col2[:,1])[0, 1]
                testr.append(rr)
                testx.append(jj-(tests-1)/2)
                testy.append(kk-(tests-1)/2)
        maxr = np.argmax(testr)
        bestx.append(int(testx[maxr]))
        besty.append(int(testy[maxr]))
        layers[:,:,ii] = np.roll(layers[:,:,ii], -bestx[ii], axis=0)
        layers[:,:,ii] = np.roll(layers[:,:,ii], -besty[ii], axis=1)
    if plot:
        plt.figure()
        plt.imshow(np.nanmean(layers, axis=2), cmap='hot')
        plt.axis('equal')
        plt.clim(7.45, 10)
        plt.axis('off')
        plt.show(block=False)
    return bestx, besty, layers

def download_fits(object_name, extension='_i2d.fits', mrp=True, include='', ptype='image'):
    os.chdir(download_fits.__code__.co_filename[:-14])
    if len(include) == 0:  # make sure include is a list
        include = []
    elif type(include) == str:
        include = [include]
    if not os.path.isdir('data'):
        os.mkdir('data')
    os.chdir('data')
    if not os.path.isdir(object_name.lower().replace(' ', '_')):
        os.mkdir(object_name.lower().replace(' ', '_'))
    os.chdir(object_name.lower().replace(' ', '_'))
    obs_table = Observations.query_object(object_name)
    if len(obs_table) == 0:
        print('No observations found for {}'.format(object_name))
        return
    obs_table = obs_table[obs_table["dataRights"] == "PUBLIC"]
    if len(ptype) > 0:
        obs_table = obs_table[obs_table["dataproduct_type"] == ptype]
    obs_table = obs_table[obs_table["obs_collection"] == "JWST"]
    to_download = []
    size = []
    for obs in obs_table:
        all = Observations.get_product_list(obs)
        filt = all[(all["productType"] == "SCIENCE") | (all["productType"] == "science")]
        filt = filt[filt["dataRights"] == "PUBLIC"]
        filt = Observations.filter_products(filt, extension=extension, mrp_only=mrp)

        if len(filt) > 0:
            if len(include) > 0:

                for jj in filt:
                    got_any = False  # when include = []
                    for inc in include:
                        if inc in jj['obs_id']:  ## looking for a dataset where filenames contain "something"
                            got_any = True
                    if got_any:
                        size.append(int(jj['size']))
                        to_download.append(jj)
    total_size = int(np.round(np.sum(size)/1e6))
    resp = 'n'
    if len(to_download) == 0:
        print('found 0 of ' + str(len(obs_table)) + ' observations')
    else:
        print('found '+str(len(to_download))+' of '+str(len(obs_table))+' observations')
        resp = input('Download {} files ({} MB) ?'.format(len(to_download), total_size))
    manifest = []
    if resp.lower() == 'y':
        for jj in to_download:
            manifest.append(Observations.download_products(jj))
    else:
        print('abort')
    return manifest

def reproject(path, project_to=0):
    # FIXME add support for exact and adaptive methods
    template = path[project_to]
    # remove the template from the path
    # path = path[:project_to] + path[project_to+1:]
    hdu_temp = fits.open(template)
    layers = np.ndarray((hdu_temp[1].shape[0], hdu_temp[1].shape[1], len(path)))
    for ii, pp in enumerate(path):
        hdu = fits.open(pp)
        if ii == project_to:
            layers[:,:,ii] = hdu[1].data
        else:
            reproj, _ = reproject_interp(hdu[1], hdu_temp[1].header)
            layers[:, :, ii] = reproj
    return layers


def clip_square_edge(shape, x0, x1, y0, y1):
    # find indices for a redctangle but clip it when before 0 or after image size
    x0 = np.max([x0, 0])
    x1 = np.min([x1, shape[0]])
    y0 = np.max([y0, 0])
    y1 = np.min([y1, shape[1]])
    return [x0, x1, y0, y1]





def get_lines(source='FS+M', lims=[4, 30]):
    '''
    Get IR emission lines from the tables listed here https://www.mpe.mpg.de/ir/ISO/linelists/index.html

    Parameters
    source: str
        which table of lines to read. default is fine structure + molecular. alternatives: 'FSlines','Molecular', 'Hydrogenic'
    lims:   list
        lower and upper wavelength limits e.g. [3,30]
     _________
     Returns
     table: DataFrame
        with species and wavelength columns (in micrometer)
    '''
    table = []
    if source == 'FSlines':
        t0 = pd.read_csv(root+'docs/FSlines.csv', sep=',')
        species = np.asarray(t0['species'])
        wavelength = np.asarray(t0['lambda'])
        # from https://www.mpe.mpg.de/ir/ISO/linelists/FSlines.html
    elif source == 'Hydrogenic':
        t0 = pd.read_csv(root+'docs/Hydrogenic.csv', sep=',', index_col=False)
        species = np.asarray(t0['ion'])
        wavelength = np.asarray(t0['vacuumwavel'])
        # https://www.mpe.mpg.de/ir/ISO/linelists/Hydrogenic.html
    elif source == 'H2':
        #https://www.mpe.mpg.de/ir/ISO/linelists/H2.html
        t0 = pd.read_csv(root+'docs/H2.csv', sep=',')
        species = np.asarray(t0['H2'])
        wavelength = t0['Wavelength']
    elif source == 'Molecular':
        # https://www.mpe.mpg.de/ir/ISO/linelists/Molecular.html
        t0 = pd.read_csv(root+'docs/Molecular.csv', sep=',')
        species = np.asarray(t0['Species'])
        wavelength = np.asarray(t0['Lambda(mu)'])
    elif source == 'FS+M':
        t1 = table = pd.read_csv(root+'docs/FSlines.csv', sep=',')
        species = np.asarray(t1['species'])
        wavelength = t1['lambda']
        t2 = pd.read_csv(root + 'docs/Molecular.csv', sep=',')
        t2['Species'] = t2['Species'].str.replace('H2','H???')
        t2['Species'] = t2['Species'].str.replace('o-H???O','H???O')
        t2['Species'] = t2['Species'].str.replace('p-H???O', 'H???O')
        t2.loc[t2['Species'].str.contains('OH'), 'Species'] = 'OH'
        species = np.concatenate([species,np.asarray(t2['Species'])])
        wavelength = np.concatenate([wavelength,np.asarray(t2['Lambda(mu)'])])
        t3 = pd.read_csv(root + 'docs/H_4to30.csv', sep=',')
        species = np.concatenate([species,np.asarray(t3['atoms'])])
        wavelength = np.concatenate([wavelength, np.asarray(t3['wavelength(um)'])])
        t4 = pd.read_csv(root + 'docs/PAH.csv', sep=',')
        species = np.concatenate([species, np.asarray(t4['label'])])
        wavelength = np.concatenate([wavelength, np.asarray(t4['wavelength'])])

    order = np.argsort(wavelength)
    species = species[order]
    wavelength = wavelength[order]
    if lims is not None:
        if len(lims) == 2:
            keep = (wavelength < lims[1]) & (wavelength > lims[0])
            species = species[keep]
            wavelength = wavelength[keep]
        else:
            raise Warning('something strange with lims, returning all')
    table = pd.DataFrame({'species': species, 'wavelength': wavelength})
    # t2['Species'] = t2['Species'].str.replace('OH1/2-3/2', 'OH')
    return table


def evaluate_redshift(flux, wavelength=None, max_z=1, resolution=0.0001, prom_med=10):
    '''
    evaluate redshift from peaks in observed data

    Parameters
    __________
    flux:       hdu | np.ndarray
        can be a hdu list, hdu BinTable (hdu[1]) or an ndarray
    wavelength: None | np.ndarray
        None if flux is hdu containing the wavelength. otherwise ndarray
    max_z: int | float
        maximum redshift to consider
    prom_med: int | float
        how many medians to consider as prominence for peak detection
    Returns
    _______
    '''
    if type(flux) == fits.hdu.hdulist.HDUList:
        wavelength = flux[1].data['WAVELENGTH']
        flux = flux[1].data['FLUX']
    elif type(flux) == fits.hdu.table.BinTableHDU:
        wavelength = flux['WAVELENGTH']
        flux = flux['FLUX']
    lims = [wavelength[0], wavelength[-1]*(max_z+1)]
    table = get_lines(lims=lims)
    w_expected = np.asarray(table['wavelength'])
    s_expected = np.asarray(table['species'])
    prom = np.median(np.abs(np.diff(flux))) * prom_med
    peaks = find_peaks(flux, prominence=prom, width=[1, 20])[0]

    best_z = 0
    err = np.inf
    # n_dec = -np.log10(resolution)
    z = -resolution
    while z < max_z:
        z += resolution
        vbest = []
        ibest = []
        for ip, peak in enumerate(peaks):
            dif = np.abs(w_expected*(1+z)-wavelength[peak])
            vbest.append(np.min(dif))
            ibest.append(np.argmin(dif))
        err0 = np.median(vbest)
        if err0 < err:
            err = err0
            best = ibest
            best_z = z
    return best_z


def filt_num(path):
    filt = np.zeros(len(path))
    for ii in range(len(path)):
        plip = path[ii][-1:0:-1]
        plip = plip.replace('-','_')
        iF = plip.find('f_')  # index of filter, sorry
        if iF == -1:
            filt[ii] = np.nan
        else:
            p = plip[:iF][-1:0:-1]
            filt[ii] = int(p[:p.find('_')-1])
    return filt


def crop_fits(hdu1, center, sizes):
    '''

    Parameters
    ----------
    hdu: hdu with hdu.data and hdu.header
    center: list, [x,y] for center of cropped rectangle
    sizes: list, [width, height]

    Returns
    -------
    hdu: cropped hdu

    '''
    wcs = WCS(hdu1.header)
    pos = wcs.wcs_pix2world([[center[0] - sizes[0] / 2, center[1] - sizes[1] / 2], center], 0)
    pix = wcs.wcs_world2pix(pos, 0)
    pix = np.round(np.asarray(pix))
    cutout = Cutout2D(hdu1.data, pix[1, :], sizes, wcs)
    hdu1.data = cutout.data
    hdu1.header.update(cutout.wcs.to_header())
    return hdu1, pos, pix



if __name__ == '__main__':
    path = list_files('/home/innereye/JWST/Ori/', search='*.fits')
    # hdu = fits.open(path[8])
    hdu = fits.open(path[8])
    # img = hdu[1].data[3800:5000, 5600:7000]
    img = hdu[1].data[1800:2400, 2800:3800]
    xy = hole_xy(img)
    size = hole_size(img, xy, plot=True)
    filled = hole_circle_fill(img, xy, size)

    fix = fill_craters(img, method='gaus')
    # fix = fill_holes(img, pad=1)
    plt.figure();plt.imshow(img[200:400,100:300]);plt.show(block=False)
    plt.figure();plt.imshow(fix[200:400,100:300]);plt.show(block=False)
    print('tada')


def movmean(data, win):
    #  smooth data with a moving average. win should be an odd number of samples.
    #  data is np.ndarray with samples by channels shape
    #  to get smoothing of 3 samples back and 3 samples forward use win=7
    if len(data.shape) == 1:
        data = data[:, np.newaxis]
        nChannels = 1
    else:
        nChannels = data.shape[1]
    smooth = data.copy()
    for iChannel in range(nChannels):
        if len(data.shape) == 1:
            vec = data
        else:
            vec = data[:, iChannel]
        padded = np.concatenate(
            (np.ones((win,)) * vec[0], vec, np.ones((win,)) * vec[-1]))
        sm = np.convolve(padded, np.ones((win,)) / win, mode='valid')
        sm = sm[int(win / 2):]
        sm = sm[0:vec.shape[0]]
        if len(data.shape) == 1:
            smooth[:] = sm
        else:
            smooth[:, iChannel] = sm
    return smooth