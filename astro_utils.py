import pandas as pd
from astropy.io import fits
from astropy import wcs
from reproject import reproject_interp
import os
# import glob
import matplotlib
# matplotlib.use('Qt5Agg')
matplotlib.use('TkAgg')
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
from bot_grabber import level_adjust
import pickle
from skimage import transform
from scipy.ndimage import label
from scipy.spatial import KDTree

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

def optimize_xy(layers, square_size=[100, 100], tests=9, plot=False):
    if square_size is None:
        szx05 = int(np.ceil(tests/2))
        szy05 = int(np.ceil(tests/2))
        square_size = np.array(layers.shape[:2]).astype(int) - tests
    else:
        szx05 = int(layers.shape[0]/2-square_size/2)
        szy05 = int(layers.shape[1]/2-square_size/2)
    square0 = layers[szx05:szx05+square_size[0], szy05:szy05+square_size[1], 0]
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
                square = layers[jitter + szx05:jitter + szx05 + square_size[0],
                                kitter + szy05:kitter + szy05 + square_size[1], ii]
                square = median_filter(square, footprint=np.ones((2, 2)))
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
        print(f'done {ii}')
    if plot:
        plt.figure()
        plt.imshow(np.nanmean(layers, axis=2), cmap='hot')
        plt.axis('equal')
        # plt.clim(7.45, 10)
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
        t2['Species'] = t2['Species'].str.replace('H2','H₂')
        t2['Species'] = t2['Species'].str.replace('o-H₂O','H₂O')
        t2['Species'] = t2['Species'].str.replace('p-H₂O', 'H₂O')
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


def auto_plot(folder='ngc1672', exp='*_i2d.fits', method='rrgggbb', pow=[1, 1, 1], pkl=True, png=False, resize=False,
              core=False, factor=4):
    # TODO: clean small holes fast without conv, remove red background
    if not os.path.isdir(folder):
        try:
            os.chdir('/home/innereye/JWST/')
            if not os.path.isdir(folder):
                raise Exception('cannot find '+folder)
        except:
            raise Exception('cannot find ' + folder)

    path = list_files(os.getcwd()+'/'+folder, exp)
    filt = filt_num(path)
    order = np.argsort(filt)
    path = np.asarray(path)[order]

    def make_rgb(prc=0):
        rgb = np.zeros((layers.shape[0], layers.shape[1], 3), float)
        for ll in range(3):
            lay = np.mean(layers[:, :, iii[ll]], axis=2)
            lay = lay ** pow[ll] * 255
            rgb[:, :, ll] = lay
        if prc > 0:  # subtract 1 percentile to remove red after **0.5
            vec = rgb[:, :, 0].copy().flatten()
            vec[vec == 0] = np.nan
            rgb[:, :, 0] = rgb[:, :, 0] - np.nanpercentile(vec, 1)
            rgb[rgb < 0] = 0
            rgb[:, :, 0] = rgb[:, :, 0] / np.nanmax(rgb[:, :, 0]) * 255
        rgb = rgb.astype('uint8')
        return rgb

    pkl_name = folder + '.pkl'
    if os.path.isfile(pkl_name) and pkl:
        layers = np.load(pkl_name, allow_pickle=True)
    else:
        for ii in range(len(path)):
            if ii == 0:
                hdu0 = fits.open(path[ii])
                img = hdu0[1].data
                if resize:# make rescale size for wallpaper 1920 x 1080
                    rat = np.max(img.shape)/np.min(img.shape)
                    if rat*9 > 16:
                        h = 1080
                        w = int(np.ceil(h*rat))
                    else:
                        w = 1920
                        h = int(np.ceil(w*(1/rat)))
                    if img.shape[0] > img.shape[1]:
                        wh = [w, h]
                    else:
                        wh = [h, w]  # rotate later
                    img = transform.resize(img, wh)
                layers = np.zeros((img.shape[0], img.shape[1], len(path)))
                hdr0 = hdu0[1].header
                hdu0.close()
            else:
                hdu = fits.open(path[ii])
                img, _ = reproject_interp(hdu[1], hdr0)
                if resize:
                    img = transform.resize(img, wh)
            layers[:, :, ii] = img
        if pkl:
            with open(pkl_name, 'wb') as f:
                pickle.dump(layers, f)
    if core:
        parts = 2
        upix = np.array(layers.shape[:2])/25
        degrade_shape = np.ceil(upix).astype('int')
        max_xy = np.zeros((layers.shape[2], 2))
        for lay in range(layers.shape[2]):
            dgd = transform.resize(layers[:,:,lay], degrade_shape)
            max_xy[lay,:] = np.unravel_index(np.nanargmax(dgd), degrade_shape)
        max_xy = np.mean(max_xy, axis=0) * 25.5
        max_xy = max_xy.astype(int)
        half = int(np.min(upix)*parts)
        layers = layers[max_xy[0] - half:max_xy[0] + half, max_xy[1] - half:max_xy[1] + half, :]
        core_str = '_core'
    else:
        core_str = ''
    for lay in range(layers.shape[2]):
        tmp = layers[:, :, lay].copy()
        mask = np.isnan(tmp)
        tmp[mask] = 0
        tmp = level_adjust(tmp, factor=factor)
        tmp[mask] = np.nan
        layers[:, :, lay] = tmp
    if method == 'rrgggbb':
        ncol = np.floor(layers.shape[-1] / 3)
        ib = np.arange(0, ncol).astype(int)
        ir = np.arange(layers.shape[-1] - ncol, layers.shape[-1]).astype(int)
        ig = np.arange(ib[-1] + 1, ir[0]).astype(int)
        iii = [ir, ig, ib]
        rgb = make_rgb()
        plt.figure()
        plt.imshow(rgb, origin='lower')
        plt.show()
    elif method == 'mnn':  # Miri Nircam Nircam
        ismiri = ['miri' in x for x in path]
        ir = np.where(ismiri)[0]
        inircam = np.where(~np.asarray(ismiri))[0]
        nb = np.ceil(len(inircam)/2)
        ib = np.arange(0, nb).astype(int)
        ig = np.arange(ib[-1] + 1, ir[0]).astype(int)
        iii = [ir, ig, ib]
        rgb = make_rgb(prc=1)
        plt.figure()
        plt.imshow(rgb, origin='lower')
        plt.show(block=False)
    if png:
        if type(png) == str:
            png_name = png
        else:
            png_name = folder+core_str+'.png'
        plt.imsave(png_name, rgb, origin='lower')


def maxima_gpt(image, neighborhood_size=10, thr=99.3, smooth=True):
    '''
    find objects as local maxima after smoothing and clustering light around star centers.
    suggested by chatGBT, modified
    Parameters
    ----------
    image : ndarray
        2D image
    neighborhood_size :
        rejects starts that are 10pix from other stars
    thr: float
        threshold for brightness to consider, in percentile units
    Returns
    -------
    maxima_image : ndarray
        2D boolean array with True for local maxima
    '''
    # find objects as clusters
    kernel = Gaussian2DKernel(3)
    if smooth:
        smoothed_image = convolve(image, kernel)
    else:
        smoothed_image = image
    # Compute the Hessian matrix
    dx, dy = np.gradient(smoothed_image)
    dxx, dxy = np.gradient(dx)
    _, dyy = np.gradient(dy)
    hessian = np.stack((np.stack((dxx, dxy), axis=-1), np.stack((dxy, dyy), axis=-1)), axis=-1)
    # Compute the eigenvalues of the Hessian matrix
    eigvals = np.linalg.eigvalsh(hessian)
    # Find candidate maxima
    # threshold = 0.5  # adjust as needed
    threshold = np.nanpercentile(image, thr)
    candidate_maxima = np.zeros(image.shape, dtype=bool)
    candidate_maxima[(eigvals[..., 0] < 0) & (eigvals[..., 1] < 0) & (image > threshold)] = True

    # Cluster candidate maxima
    clustered_maxima, num_clusters = label(candidate_maxima)
    maxima_locations = np.array(np.where(candidate_maxima))
    maxima_intensities = image[candidate_maxima]
    if num_clusters > 5000:
        raise Exception(f'{num_clusters} clusters!')
    else:
        print(f'{num_clusters} clusters')
    # Keep only highest intensity maximum in each cluster
    filtered_maxima = np.zeros(image.shape, dtype=bool)
    for i in range(1, num_clusters + 1):
        cluster_mask = (clustered_maxima == i)
        cluster_intensities = np.zeros(image.shape)
        cluster_intensities[cluster_mask] = smoothed_image[cluster_mask]  # maxima_intensities[cluster_mask]
        max_index = np.argmax(cluster_intensities)
        max_location = np.unravel_index(max_index, image.shape)
        filtered_maxima[max_location[0], max_location[1]] = True

    # Filter out maxima closer than 10 pixels apart
    if neighborhood_size is None:
        maxima_image = filtered_maxima
    else:
        maxima_image = np.zeros(image.shape, dtype=bool)
        maxima_coords = np.argwhere(filtered_maxima)
        for coord in maxima_coords:
            y, x = coord
            neighborhood = filtered_maxima[max(0, y - neighborhood_size):min(image.shape[0], y + neighborhood_size + 1),
                           max(0, x - neighborhood_size):min(image.shape[1], x + neighborhood_size + 1)]
            if np.sum(neighborhood) == 1:
                maxima_image[y, x] = True
    return maxima_image


def optimize_xy_clust(layers, smooth=True, neighborhood_size=10, thr=99.3, plot=False):

    maxima_xy = []
    for lay in range(layers.shape[2]):
        max_image = maxima_gpt(layers[:, :, lay], smooth=smooth, neighborhood_size=neighborhood_size, thr=thr)
        maxima_xy.append(np.array(np.where(max_image)).T)
    # bestx = [0]
    # besty = [0]
    order = np.argsort([len(x) for x in maxima_xy])
    bestx = [[]]*len(maxima_xy)
    besty = [[]]*len(maxima_xy)
    bestx[order[0]] = 0
    besty[order[0]] = 0
    for ii in order[1:]:  # range(1, layers.shape[2]):
        tree = KDTree(maxima_xy[ii])
        distances, closest_points = tree.query(maxima_xy[order[0]])
        if plot:
            from cv2 import line  # conflict with %matplotlib qt, import only if used
            plt.figure()
            tmp = layers.copy()
            for jj in range(len(maxima_xy[order[0]])):
                y1, x1 = maxima_xy[ii][closest_points[jj], :]
                y2, x2 = maxima_xy[order[0]][jj, :]
                line(tmp, (x1, y1), (x2, y2), (0, layers.max(), 0), thickness=1)
            plt.imshow(tmp)
            del tmp
        # d = plt.hist(distances, np.arange(100))
        count, bins = np.histogram(distances, np.arange(100))  # , normed=True)
        common = np.argmax(count)   # common distance between stars in layer a and b
        # Find nearest neighbor in second set for each point in first set
        xx = maxima_xy[order[0]][:, 0]
        xshifts = maxima_xy[ii][closest_points, 0] - xx
        yy = maxima_xy[order[0]][:, 1]
        yshifts = maxima_xy[ii][closest_points, 1] - yy
        try:
            bestx[ii] = int(np.median(maxima_xy[ii][closest_points[(common-2 < distances) & (distances < common+2)], 0] - maxima_xy[order[0]][(common-2 < distances) & (distances < common+2), 0]))
        except:
            bestx[ii] = 999
        try:
            besty[ii] = int(np.median(maxima_xy[ii][closest_points[(common-2 < distances) & (distances < common+2)], 1] - maxima_xy[order[0]][(common-2 < distances) & (distances < common+2), 1]))
        except:
            besty[ii] = 999
        if (bestx[ii] != 0) and (bestx[ii] != 999):
            layers[:, :, ii] = np.roll(layers[:, :, ii], -bestx[ii], axis=0)
        if (besty[ii] != 0) and (besty[ii] != 999):
            layers[:, :, ii] = np.roll(layers[:, :, ii], -besty[ii], axis=1)
    return bestx, besty, layers


if __name__ == '__main__':
    # auto_plot('ngc3256', '*w_i2d.fits', method='mnn')
    os.chdir('/home/innereye/JWST/ngc5068/')
    layers = np.load('ngc5068.pkl', allow_pickle=True)
    xstart = 3820
    ystart = 3300
    crop = layers.copy()[xstart:xstart + 500, ystart:ystart + 500, :3]
    for lay in range(3):
        crop[:,:,lay] = level_adjust(crop[:,:,lay])
    bestx, besty, _ = optimize_xy_clust(crop, smooth=True, plot=True, neighborhood_size=None, thr=90)

    # os.chdir('/home/innereye/JWST/')
    # rut = '/home/innereye/JWST/SDSSJ1723+3411/MAST_2022-08-31T1707/JWST/'
    # path = [rut+'/jw01355-o010_t009_miri_f560w/jw01355-o010_t009_miri_f560w_i2d.fits',
    #         rut+'/jw01355-o009_t009_nircam_clear-f444w/jw01355-o009_t009_nircam_clear-f444w_i2d.fits',
    #         rut+'jw01355-o009_t009_nircam_clear-f277w/jw01355-o009_t009_nircam_clear-f277w_i2d.fits']
    # layers = reproject(path, project_to=0)
    # for lay in range(3):
    #     layer = layers[:,:,lay]
    #     mask = np.isnan(layer)
    #     layer[mask] = 0
    #     layer = level_adjust(layer)
    #     layer[mask] = np.nan
    #     layers[:, :, lay] = layer
    # layers = layers**2  # hide background noise
    # layers[:,:,1:] = layers[:,:,1:]**2  # stronger red
    # layers = layers[165:, 385:-35, :]
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.imshow(layers)
    # # layers.orig = layers.copy()
    # # plt.imshow(layers)
    # bestx, besty, layers = optimize_xy_clust(layers)
    # bestx, besty, layers = optimize_xy(layers, square_size=None, tests=9, plot=False)
    # plt.subplot(1, 2, 2)
    # plt.imshow(layers)
    # # print('start')
    # # auto_plot('ngc1512', '*_i2d.fits', png=True, pow=[0.5, 1, 1], resize=True)
    # # auto_plot('ngc1672', '*_i2d.fits', png='core1.png', pow=[1, 1, 1], core=True)
    # # auto_plot('ngc1672', '*_i2d.fits', png='core05.png', pow=[0.5, 1, 1], core=True)
    # # auto_plot('ngc1672', '*_i2d.fits', png='red_sqrt.png', method='mnn', pow=[0.5, 1, 1], pkl=True, factor=2)
    # # auto_plot('ngc1672', '*_i2d.fits', png=False)
    # print('tada')
