import pandas as pd
from astropy.io import fits
from astropy import wcs
from reproject import reproject_interp
import os
from glob import glob
import matplotlib
# matplotlib.use('Qt5Agg')
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from astropy.convolution import Ring2DKernel, Gaussian2DKernel, convolve
from astropy.time import Time
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from scipy.ndimage import median_filter, maximum_filter
from scipy.signal import find_peaks, medfilt
from astroquery.mast import Observations
# from skimage.morphology import disk
from bot_grabber import level_adjust, nanmask, get_JWST_products_from
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
    # os.chdir(root+'data')
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
        szx05 = int(layers.shape[0]/2-square_size[0]/2)
        szy05 = int(layers.shape[1]/2-square_size[1]/2)
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


def download_fits_files(file_names, destination_folder='', overwrite=False):
    if len(destination_folder) > 0 and destination_folder[-1] not in '\/':
        destination_folder += '/'
    mast = 'https://mast.stsci.edu/portal/Download/file/JWST/product/'
    no_print = '>/dev/null 2>&1'
    success = 0
    for fn in file_names:
        fn = fn.split('/')[-1]
        if not os.path.isfile(destination_folder+fn) or overwrite:
            a = os.system(f'wget -O {destination_folder+fn} {mast}{fn} {no_print}')
            if a == 0:
                success += 1
    print(f'Downloaded {success} files to {destination_folder}')

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
              core=False, plot=True, factor=4, smooth=False, crop=False):
    '''
    finds fits files in path according to expression exp, and combine them to one RGB image.
    Parameters
    ----------
    folder: str
        All fits files in this directory and subdirectories will be combined, presuming they fit exp
    exp: str | list
        regexp expression to filter files by their name. by default, take all *_i2d.fits
        if exp is a list, it takes it as path to specific fits files inside "folder"
    method: str
        the method for combining many filters into 3 RGB layers
        'rrgggbb': try divide the filters to three equal size groups. The middle group might be bigger
        'mnn': red is averaged MIRI, NIRCam images are split between green and blue
        'mtn': red is MIRI, green is total light, blue is NirCam
        'filt': assign colors from jet colormap according to filter frequency.
    pow: [float, float, float]
        A list of 3 numbers by which to rise power of the rgb image. Power is computed for rgb between 0 and 1, so
        using [0.5, 1,  1] will increase visibility of low light red pixels. [1,1,1] means no action.
    pkl: bool | str
        True for save nd array of all data as pickle first time this directory is processed, and read it if it exists next times
        str means True + pkl filename to read / save
    png: bool | str
        False - don't save png. True - save png according to folder name. str - specify png name to save.
    resize: bool
        try resize to fit a 1920 by 1080 image. no cropping or aspect artio chabges. meant to reduce RAM and time.
    core: bool
        True to try focus on the core of the galaxy, in order to stretch the colors differently.
    factor: int
        this got to do with color stretching, usually should't be touched

    Returns
    -------
    rgb: np.ndarray

    '''
    # TODO: clean small holes fast without conv, remove red background
    for search in ['/media/innereye/My Passport/Data/JWST/data/',
                   './',
                   '../',
                   '/home/innereye/astro/data/',
                   '/home/innereye/JWST/']:
        if os.path.isdir(search+folder):
            os.chdir(search)
            break
    if not os.path.isdir(folder):
        raise Exception('cannot find '+folder)
    if type(exp) == str:
        if exp[:3] == 'log':
            if len(exp) > 3:
                logpath = os.environ['HOME'] + '/astro/logs/' + exp.replace('log','')
                log = pd.read_csv(logpath)
            else:
                logpath = glob(os.environ['HOME'] + '/astro/logs/' + folder + '*')
                if len(logpath) == 1:
                    log = pd.read_csv(logpath[0])
                else:
                    print(logpath)
                    raise Exception('expextec one log file')
            path = list(log['file'][log['chosen']])
            os.chdir(folder)
        else:
            path = list_files(os.getcwd()+'/'+folder, exp)
    else:
        path = exp
        os.chdir(folder)
    filt = filt_num(path)
    order = np.argsort(filt)
    path = np.asarray(path)[order]
    filt = filt[order]
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

    def resize_wh(shape):
        rat = np.max(shape) / np.min(shape)
        if rat * 9 > 16:
            h = 1080
            w = int(np.ceil(h * rat))
        else:
            w = 1920
            h = int(np.ceil(w * (1 / rat)))
        if shape[0] > shape[1]:
            wh = [w, h]
        else:
            wh = [h, w]  # rotate later
        return wh
    pkl_name = folder + '.pkl'
    if type(pkl) == str:
        pkl_name = pkl
        pkl = True
    if os.path.isfile(pkl_name) and pkl:
        layers = np.load(pkl_name, allow_pickle=True)
        if len(path) < layers.shape[2]:
            os.chdir(search)
            path_full = list_files(os.getcwd()+'/'+folder, '*_i2d.fits')
            if len(path_full) == layers.shape[2]:
                filt_full = filt_num(path_full)
                order_full = np.argsort(filt_full)
                path_full = np.asarray(path_full)[order_full]
                include = [int(np.where(path_full == x)[0][0]) for x in path]
                layers = layers[:,:,include]
            else:
                raise Exception('which layer is which file?')
        if resize:
            wh = resize_wh(layers.shape[:2])
            layers = transform.resize(layers, wh)
    else:
        for ii in range(len(path)):
            if ii == 0:
                hdu0 = fits.open(path[ii])
                img = hdu0[1].data
                if resize:# make rescale size for wallpaper 1920 x 1080
                    wh = resize_wh(img.shape)
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
            if os.path.isfile(pkl_name):
                print('not saving pickle')
            else:
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
    elif crop:
        if type(crop) == str:
            if 'y1' in crop:
                ldict = {}
                exec(crop, globals(), ldict)
                x1 = ldict['x1']
                x2 = ldict['x2']
                y1 = ldict['y1']
                y2 = ldict['y2']
                print(y1)
            else:
                raise Exception('I expect string to be like "y1=1003; y2=2248; x1=949; x2=1926"')
        else:
            lay3 = [0, int(layers.shape[2]/2), layers.shape[2]-1]
            imtochoose = layers[..., lay3].copy()
            for i3 in range(3):
                imtochoose[..., i3] = level_adjust(imtochoose[..., i3])
            plt.figure()
            plt.imshow(level_adjust(imtochoose))
            plt.axis('off')
            click_coordinates = []
            def onclick(event):
                if len(click_coordinates) < 2:
                    # Append the coordinates of the clicked point to the list
                    click_coordinates.append((event.xdata, event.ydata))
                    # Plot a red dot at the clicked point
                    plt.plot(event.xdata, event.ydata, 'ro')
                    plt.draw()
                    if len(click_coordinates) == 2:
                        # After collecting two points, close the figure to proceed
                        plt.close()
            plt.connect('button_press_event', onclick)
            plt.show()
            p1, p2 = click_coordinates
            x1, y1 = int(min(p1[0], p2[0])), int(min(p1[1], p2[1]))
            x2, y2 = int(max(p1[0], p2[0])), int(max(p1[1], p2[1]))
            print(f'y1={y1}; y2={y2}; x1={x1}; x2={x2}')
        layers = layers[y1:y2, x1:x2]
        core_str = '_crop'
    else:
        core_str = ''
    empty = np.zeros(len(path), bool)
    for lay in range(layers.shape[2]):
        if np.mean(np.isnan(layers[:, :, lay])) == 1 or layers[:, :, lay].sum() == 0:
            empty[lay] = True
        else:
            layers[:, :, lay] = level_adjust(layers[:, :, lay], factor=factor)
            if smooth:
                layers[:, :, lay] = smooth_yx(layers[:, :, lay], 5, 2)
    # combine colors by method
    layers = layers[..., ~empty]
    path = path[~empty]
    rgb = None
    if method == 'rrgggbb':
        ncol = np.floor(layers.shape[-1] / 3)
        ib = np.arange(0, ncol).astype(int)
        ir = np.arange(layers.shape[-1] - ncol, layers.shape[-1]).astype(int)
        ig = np.arange(ib[-1] + 1, ir[0]).astype(int)
        iii = [ir, ig, ib]
    elif method == 'mnn':  # Miri Nircam Nircam
        ismiri = ['miri' in x for x in path]
        ir = np.where(ismiri)[0]
        inircam = np.where(~np.asarray(ismiri))[0]
        nb = np.ceil(len(inircam)/2)
        ib = np.arange(0, nb).astype(int)
        ig = np.arange(ib[-1] + 1, ir[0]).astype(int)
        iii = [ir, ig, ib]
    elif method == 'mtn':  # Miri Total NirCam
        ismiri = ['miri' in x for x in path]
        ir = np.where(ismiri)[0]
        ib = np.where(~np.asarray(ismiri))[0]
        ig = np.arange(layers.shape[2]).astype(int)
        iii = [ir, ig, ib]
    elif method == 'filt':
        col = matplotlib.cm.jet(filt / np.max(filt))[:, :3]  # [:, ::-1]
        rgb = assign_colors(layers, col)
        # rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
        # for lay in range(layers.shape[2]):
        #     for ic in range(3):
        #         rgb[:, :, ic] = np.nanmax(
        #             np.array([rgb[:, :, ic], layers[:, :, lay] * col[lay, ic]]), 0)
        #         # rgb[:, :, ic] = np.nansum(
        #         #     np.array([rgb[:, :, ic], layers[:, :, lay] * (1 / layers.shape[2]) * col[lay, ic]]), 0)
        for ic in range(3):
            rgb[:, :, ic] = rgb[:, :, ic] ** pow[ic] * 255
        rgb = rgb.astype('uint8')
    if rgb is None:
        rgb = make_rgb()
    if plot:
        plt.figure()
        plt.imshow(rgb, origin='lower')
        plt.show()
    if png:
        if type(png) == str:
            png_name = png
        else:
            png_name = folder+core_str+'.png'
        plt.imsave(png_name, rgb, origin='lower')
    return rgb


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


def optimize_xy_manual(layers):
    '''
    A GUI for nudging images so they overlap well. after computing xy, use roll to align layers.
    Exaample:
    xy = optimize_xy_manual(crop)
    shifted = roll(crop, xy, nan_edge=True)
    Parameters
    ----------
    layers

    Returns
    -------

    '''
    img = np.zeros((layers.shape[0], layers.shape[1], 3))
    xy = [[0, 0]]
    fig = plt.figure()# first layer don't move
    for cur in range(1, layers.shape[2]):
        img[:, :, 2] = layers[:, :, 0]
        orig = layers[:, :, cur].copy()
        img[:, :, 0] = orig.copy()
        subKey = ['']
        def subname(event):
            subKey[0] = event.key

        # plt.pause(0.3)
        connectID = fig.canvas.mpl_connect('key_press_event', subname)
        im = plt.imshow(img, origin='lower')
        xy.append([0, 0])
        plt.title(f'push red with arrows\nlayer {cur}/{layers.shape[2]}, shift: {xy[-1]}', fontsize=18)
        plt.draw()
        keepTyping = True

        while keepTyping:
            plt.waitforbuttonpress()
            if subKey[0] == 'enter':
                keepTyping = False
            else:
                if subKey[0] == 'left':
                    xy[-1][1] = xy[-1][1] - 1
                    img[:, :, 0] = np.roll(img[:, :, 0], -1, axis=1)
                elif subKey[0] == 'right':
                    xy[-1][1] = xy[-1][1] + 1
                    img[:, :, 0] = np.roll(img[:, :, 0], 1, axis=1)
                elif subKey[0] == 'down':
                    xy[-1][0] = xy[-1][0] - 1
                    img[:, :, 0] = np.roll(img[:, :, 0], -1, axis=0)
                elif subKey[0] == 'up':
                    xy[-1][0] = xy[-1][0] + 1
                    img[:, :, 0] = np.roll(img[:, :, 0], +1, axis=0)
            print(subKey[0])
            im.set_array(img)
            plt.title(f'push red with arrows\nlayer {cur}/{layers.shape[2]}, shift: {xy[-1]}', fontsize=18)
            plt.draw()
    fig.canvas.mpl_disconnect(connectID)
    xy = np.array(xy)
    return xy


def roll(a, shift, nan_edge=True):
    '''
    np.roll while filling nans for edge pixel overflow
    should be used after optimize_xy_manual.
    Parameters
    ----------
    a: 2D or 3D np.ndarray
    shift: how much to roll for x and y, per layer
    nan_edge: True to fill nans where pixels were shifted from

    Returns
    -------
    a: the shifted np.array
    '''
    shift = shift
    if len(a.shape) == 2:
        a = a[..., np.newaxis]
    for lay in range(a.shape[2]):
        a[..., lay] = np.roll(a[..., lay], shift[lay, :], axis=[0, 1])
        if nan_edge:
            if shift[lay, 1] > 0:
                a[:, :shift[lay, 1], lay] = np.nan
            elif shift[lay, 1] < 0:
                a[:, shift[lay, 1]:, lay] = np.nan
            if shift[lay, 0] > 0:
                a[:shift[lay, 0], :, lay] = np.nan
            elif shift[lay, 0] < 0:
                a[shift[lay, 0]:, :, lay] = np.nan
    a = np.squeeze(a)
    return a

def assign_colors(images, colors):
    """
    Combine a list of grayscale images into a single RGB image by assigning each image a color from the provided list.
    Author - chatGPT

    Args:
        images (np.array): A 3D numpy array of grayscale images
        colors (list): A list of RGB color values to assign to each image in the composite image. The length of this list must
            match the number of images in the input `images` array.

    Returns:
        np.array: A 3D numpy array representing the composite RGB image.
    """
    images, _ = nanmask(images)
    # Create an empty array for the combined RGB image
    rgb_arr = np.zeros((images.shape[0], images.shape[1], 3))
    # Assign colors to each pixel of the combined RGB image based on the grayscale values of the input images
    for i in range(images.shape[2]):
        rgb_arr[:, :, 0] += images[:, :, i] * colors[i][0]
        rgb_arr[:, :, 1] += images[:, :, i] * colors[i][1]
        rgb_arr[:, :, 2] += images[:, :, i] * colors[i][2]
    # Normalize the RGB values to the range [0, 1]
    rgb_arr /= np.max(rgb_arr)
    return rgb_arr


def smooth_yx(img, win=5, passes=2):
    '''
    smooth image by running medfilt on y axsis, then on x.
    Parameters
    ----------
    img : 2D ndarray
    win : odd integer
        window for median filter
    passes : int
       number of smoothing to do

    Returns
    -------
    smooth : 2D ndarray
    '''
    if type(passes) == int:
        passes = list(range(passes))
    smooth = img.copy()
    for pas in passes:
        imgx = smooth.copy()
        for ii in range(img.shape[1]):
            vec = medfilt(imgx[:, ii], win)
            imgx[:, ii] = vec
        imgyx = imgx.copy()
        for ii in range(img.shape[0]):
            vec = medfilt(imgyx[ii, :], win)
            imgyx[ii, :] = vec
        imgy = smooth.copy()
        for ii in range(img.shape[0]):
            vec = medfilt(imgy[ii, :], win)
            imgy[ii, :] = vec
        img2 = np.array([imgx, imgy]).min(0)
        smooth = img2
    return smooth

def smooth_colors(img):
    shift = 0.000000000001
    # img = np.swapaxes(img, 0,1)
    for row in range(img.shape[0]):
        vec = img[row, :, :].copy()
        vec[np.isnan(vec)] = 0
        vec = vec + shift  # prevent nans
        rank = np.argsort(-vec, 1)
        bad_rank = [vec[ii, rank[ii, 0]] / vec[ii, rank[ii, 1]] for ii in range(len(vec))]
        bad_rank = np.array(bad_rank)
        large_peak = np.ones(len(vec))
        for ii in range(1, len(vec) - 1):
            large_peak[ii] = vec[ii, rank[ii, 0]] / np.max([vec[ii - 1, rank[ii, 0]] + vec[ii + 1, rank[ii, 0]]])
        vecc = vec.copy()
        for jj in np.where(bad_rank > 1.5)[0]:
            if jj < len(vec)-1:
                # vecc[jj, rank[jj, 0]] = (vec[jj - 1, rank[jj, 0]] + vec[jj + 1, rank[jj, 0]]) / 2
                # vecc[jj, rank[jj, 0]] = vec[jj - 1, rank[jj, 1]]
                vecc[jj, rank[jj, 0]] = np.median(vec[jj - 1, :])
        img[row,:,:] = vecc - shift
        # print(row)
    return img


def rgb2cmyk(rgb):
    cmyk_scale = 255
    if rgb.max() > 1:
        rgb = rgb.astype(float) / 255.
    K = 1 - np.max(rgb, axis=2)
    C = (1 - rgb[..., 0] - K) / (1 - K)
    M = (1 - rgb[..., 1] - K) / (1 - K)
    Y = (1 - rgb[..., 2] - K) / (1 - K)
    cmyk = (np.dstack((C, M, Y, K)) * cmyk_scale)
    return cmyk

def cmyk2rgb(cmyk, cmyk_scale=255, rgb_scale=255):
    rgb = np.zeros((cmyk.shape[0], cmyk.shape[1], 3))
    for lay in range(3):
        rgb[..., lay] = rgb_scale * (1.0 - cmyk[..., lay] / float(cmyk_scale)) * (1.0 - cmyk[..., 3] / float(cmyk_scale))
    return rgb


def last_n_days(n=3, html=True, products=False):
    end_time = Time.now().mjd
    start_time = end_time - n
    if products:
        table = get_JWST_products_from(start_time=start_time, end_time=end_time)
        suf = '_products'
    else:
        # table = Observations.query_criteria(obs_collection="JWST",
        #                                         instrument_name=["NIRCAM", "MIRI", "NIRCAM/IMAGE", "MIRI/IMAGE", "NIRISS/IMAGE"],
        #                                         t_min=[start_time, end_time],
        #                                         calib_level=3,
        #                                         dataproduct_type="image")
        table = Observations.query_criteria(obs_collection="JWST",
                                                t_obs_release=[start_time, end_time],
                                                calib_level=3,
                                                dataproduct_type="image")
        if len(table) > 0:
            table = table[table['intentType'] == 'science']
            if len(table) > 0:
                table = table[table['dataRights'] == 'PUBLIC']
                if len(table) > 0:
                    tdf = table.to_pandas()
                    table = table[list(~tdf['obs_title'].str.contains("alibration"))]
        suf = ''
    if html:
        if len(table) > 0:
            page = '<!DOCTYPE html>\n<html>\n<head>\n  <title>Image Display Example</title>\n  <style>\n   img {\n      max-width: 19vw; /* Limit image width to P% of viewport width */\n      height: auto; /* Maintain aspect ratio */\n    }\n  </style>\n</head>\n<body>'
            for iimg in range(len(table)):
                if products:
                    jpg = table['productFilename'][iimg].replace('_i2d.fits', '_i2d.jpg')
                    desc = table['description'][iimg]
                else:
                    jpg = table['jpegURL'][iimg].replace('mast:JWST/product/', '')
                    desc = 'title: '+table['obs_title'][iimg]+'\n'+'proposal: '+table['proposal_id'][iimg]+'\n'+jpg
                page = page + '\n<img src="https://mast.stsci.edu/portal/Download/file/JWST/product/' + \
                              jpg + f'" title="{desc}">'
        else:
            page = page + '\n<img src="https://hiredhandshomecare.com/wp-content/uploads/2016/10/iStock_87766021_sad-face_600px-wide.jpg" title="nothing new">'
        page = page + '\n</body>\n</html>\n'
        with open('docs/news'+suf+'.html', "w") as text_file:
            text_file.write(page)
    return table

def last_100(html=True, products=False):
    end_time = Time.now().mjd
    start_time = end_time - n
    table = Observations.query_criteria(obs_collection="JWST",
                                            t_obs_release=[start_time, end_time],
                                            calib_level=3,
                                            dataproduct_type=["image"])
    if len(table) > 0:
        table = table[table['intentType'] == 'science']
        if len(table) > 0:
            table = table[table['dataRights'] == 'PUBLIC']
            if len(table) > 0:
                tdf = table.to_pandas()
                table = table[list(~tdf['obs_title'].str.contains("alibration"))]

    if html:
        if len(table) > 0:
            page = '<!DOCTYPE html>\n<html>\n<head>\n  <title>Image Display Example</title>\n  <style>\n   img {\n      max-width: 19vw; /* Limit image width to P% of viewport width */\n      height: auto; /* Maintain aspect ratio */\n    }\n  </style>\n</head>\n<body>'
            for iimg in range(len(table)):
                if products:
                    jpg = table['productFilename'][iimg].replace('_i2d.fits', '_i2d.jpg')
                    desc = table['description'][iimg]
                else:
                    jpg = table['jpegURL'][iimg].replace('mast:JWST/product/', '')
                    desc = 'title: '+table['obs_title'][iimg]+'\n'+'proposal: '+table['proposal_id'][iimg]+'\n'+jpg
                page = page + '\n<img src="https://mast.stsci.edu/portal/Download/file/JWST/product/' + \
                              jpg + f'" title="{desc}">'
        else:
            page = page + '\n<img src="https://hiredhandshomecare.com/wp-content/uploads/2016/10/iStock_87766021_sad-face_600px-wide.jpg" title="nothing new">'
        page = page + '\n</body>\n</html>\n'
        with open('docs/news'+suf+'.html', "w") as text_file:
            text_file.write(page)
    return table


if __name__ == '__main__':
    # auto_plot('ngc3256', '*w_i2d.fits', method='mnn')
    auto_plot('ORIBAR-IMAGING-NIRCAM', exp='*_cle*.fits', png='clear.png', pow=[1, 1, 1], pkl=False, crop=True,
              method='rrgggbb')
    auto_plot('NGC-7469', exp='logNGC-7469_2022-07-01.csv', png=False, pow=[1, 1, 1], pkl=False, resize=True, method='mnn', plot=True)
