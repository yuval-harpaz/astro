from astro_deband import deband_layer
import pandas as pd
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import astropy.units as u
from astroquery.simbad import Simbad
from astroquery.mast import Observations
from reproject import reproject_interp
from glob import glob
import matplotlib
import numpy as np
from pathlib import Path
from scipy.ndimage import median_filter
from scipy.signal import medfilt
from bot_grabber import level_adjust, nanmask, get_JWST_products_from
import pickle
from skimage import transform, img_as_ubyte
from scipy.ndimage import label
from scipy.spatial import KDTree
from astro_fill_holes import *
import networkx as nx

root = os.environ['HOME']+'/astro/'
mast_url = 'https://mast.stsci.edu/portal/Download/file/JWST/product/'
def list_files(parent, search='*_i2d.fits', exclude='', include=''):
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


def fix_wcs_by_log(p, log, hdul=None):
    '''

    Args:
        p: filename
        hdul: hdu = fits.open(p)
        log: df with CRVAL1fix

    Returns:
        hdu with fixed CRVAL1 and CRVAL2
    '''
    if hdul is None:
        hdu = fits.open(p)
    if type(log) == str:
        log = pd.read_csv(log)
    row = np.where(log['file'] == p)[0]
    if len(row) == 1:
        row = row[0]
        if 'CRVAL1fix' in log.columns:
            new = log.iloc[row]['CRVAL1fix']
            if ~np.isnan(new):
                hdul[1].header['CRVAL1'] = new
            new = log.iloc[row]['CRVAL2fix']
            if ~np.isnan(new):
                hdul[1].header['CRVAL2'] = new
        else:
            raise Exception('no CRVAL1fix in log')
    else:
        raise Exception(f'expected one log row with {p}')
    return hdul

def mosaic_xy(path, plot=False, log=None):
    '''
    get x y origins of each image
    path: list of file names
    plot: bool
    log: df with coordinates for fixing the original
    :return:
    '''
    crval = np.empty((len(path), 3))
    crval[:, :] = np.nan
    dims = np.ndarray((len(path), 1))
    size = []
    pix = []
    for ii, p in enumerate(path):
        hdul = fits.open(p)
        if log is not None:
            hdul = fix_wcs_by_log(p, log, hdul=hdul)
            # row = np.where(log['file'] == p)[0]
            # if len(row) == 1:
            #     row = row[0]
            #     if 'CRVAL1fix' in log.columns:
            #         new = log.iloc[row]['CRVAL1fix']
            #         if ~np.isnan(new):
            #             hdul[1].header['CRVAL1'] = new
            #         new = log.iloc[row]['CRVAL2fix']
            #         if ~np.isnan(new):
            #             hdul[1].header['CRVAL2'] = new
            #     else:
            #         raise Exception('no CRVAL1fix in log')
            # else:
            #     raise Exception(f'expected one log row with {p}')

        w = wcs.WCS(hdul[1].header)
        if ii == 0:
            w0 = w.copy()
            pix_size0 =  hdul[1].header['CDELT1']
        pix_size = hdul[1].header['CDELT1']
        if abs(1-pix_size/pix_size0) > 0.1:
            raise Exception('pixel size more than 10% different')
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

##
def mosaic(data, xy=[], size=[], clip=[], method='overwrite', plot=False, log=None, fill=False, subtract=None):
    '''
    make a mosaic of fits images with the same pixel size
    Args:
        data: a list of file names
        xy: op of mosaic_xy
        size: op of mosaic_xy
        clip:
        method: 'mean' | 'layers' | 'median' | 'overwrite'
        plot: bool
        log: use log to fix positions

    Returns:

    '''
    if len(data) < 2:
        raise Exception('need at least 2 images')
    if len(xy) == 0:
        if (type(data[0]) == str) | (type(data[0]) == np.str_):
            xy, size = mosaic_xy(data, plot=plot, log=log)
    if (type(data[0]) == str) | (type(data[0]) == np.str_):
        path = data
        data = []
        for idat, p in enumerate(path):
            hdul = fits.open(p)
            hdudata = hdul[1].data.copy()
            if subtract:
                key = [s for s in subtract.keys() if s in p]
                if len(key) == 1:
                    hdub = fits.open(subtract[key[0]])
                else:
                    print(key)
                    raise Exception('failed to recognize noise data')
                if hdudata.shape == hdub[1].data.shape:
                    hdudata = hdudata - hdub[1].data
                    # hdudata[hdudata < 0] = 0
                else:
                    print('wrong size for '+p+' '+str(hdub[1].data.shape))

            data.append(hdudata)
            if fill:
                data[-1] = hole_func_fill(data[-1])
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
##
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


def download_fits_files(file_names, destination_folder='', overwrite=False, wget=True):
    if type(file_names) == str:
        file_names = [file_names]
    if len(destination_folder) > 0 and destination_folder[-1] not in '\/':
        destination_folder += '/'
    if not os.path.isdir(destination_folder):
        os.system('mkdir '+destination_folder)

    no_print = '>/dev/null 2>&1'
    success = 0
    for fn in file_names:
        fn = fn.split('/')[-1]
        dfn = destination_folder+fn
        if not os.path.isfile(dfn) or overwrite or os.path.getsize(dfn) == 0:
            if wget:
                a = os.system(f'wget -O {dfn} {mast_url}{fn} {no_print}')
                if a == 0:
                    success += 1
            else:
                try:
                    with fits.open(mast_url+fn, use_fsspec=True) as hdul:
                        hdr0 = hdul[0]
                        hdr = hdul[1].header
                        img = hdul[1].data
                    hdulist = fits.HDUList()
                    hdu = fits.ImageHDU()
                    hdu.data = img.copy()
                    hdu.header = hdr.copy()
                    hdulist.append(hdr0.copy())
                    hdulist.append(hdu)
                    hdulist.writeto(dfn, overwrite=True)
                    success += 1
                except Exception as error:
                    print(error)
    print(f'Downloaded {success} files to {destination_folder}')


def download_obs(df='tmp_new.csv', stingy=True):
    if type(df) == str:
        df = pd.read_csv(df)
    tgt_list = list(np.unique(df['target_name']))
    results = pd.DataFrame(tgt_list, columns=['target'])
    results['size'] = 0
    for itgt in range(len(tgt_list)):
        tgt = tgt_list[itgt]
        df1 = df[df['target_name'] == tgt]
        filt = filt_num(list(df1['dataURL']))
        order = np.argsort(filt)
        df1 = df1.iloc[order]
        df1 = df1.reset_index()
        if stingy:
            # size_ok = True
            for ifile in range(len(df1)):
                prod = Observations.get_product_list(str(df1['obsid'][ifile]))
                fits_name = df1['dataURL'][ifile]
                prodrow = np.where(prod['dataURI'] == fits_name)[0]
                if len(prodrow) == 0:
                    raise Exception('unable to find '+fits_name)
                if len(prodrow) > 1:
                    raise Exception('too many rows for '+fits_name)
                else:
                    prodrow = prodrow[0]
                sizeMB = prod['size'][prodrow]/10**6
                tot_size += sizeMB
            print(f'{tot_size}MB for {tgt} {fits_name}')
            results.at[itgt, 'size'] = tot_size
        if results['size'][itgt] < 3000:  # less than 3000MB
            for ifile in range(len(df1)):
                fits_name = df1['dataURL'][ifile]
                download_fits_files(fits_name, destination_folder='data/'+tgt)
    return results



def reproject(path, project_to=0, log=None):
    # FIXME add support for exact and adaptive methods
    template = path[project_to]
    # remove the template from the path
    # path = path[:project_to] + path[project_to+1:]
    hdu_temp = fits.open(template)
    if log is not None:
        hdu_temp = fix_wcs_by_log(template, log, hdul=hdu_temp)
    layers = np.ndarray((hdu_temp[1].shape[0], hdu_temp[1].shape[1], len(path)))
    for ii, pp in enumerate(path):
        hdu = fits.open(pp)
        if log is not None:
            hdu = fix_wcs_by_log(pp, log, hdul=hdu)
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
            inn = 0  # is not a number
            for ip in p:
                if ip.isnumeric():
                    inn += 1
                else:
                    break
            filt[ii] = int(p[:inn])
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


def blc_image(img):
    if img.dtype == 'uint8':
        uint8 = True
        img = img.astype(float)
    else:
        uint8 = False
    for ii in range(3):
        layer = img[..., ii]
        layer[layer <= 0] = np.nan
        layer = layer - np.nanmin(layer)
        layer = layer / np.nanmax(layer)
        layer[np.isnan(layer)] = 0
        img[..., ii] = layer
    if uint8:
        img = img*255
        img = img.astype('uint8')
    return img



def add_time(spaced, h_d):
    # Add h m s for RA and DEC coordinates from simbad
    toadd = [h_d, 'm', 's']
    spaced = spaced.split(' ')
    timed = ''
    for seg in range(len(spaced)):
        timed += spaced[seg]+toadd[seg]
    return timed


def crop_xy(crop):
    # Get x and y values to crop from string input
    if 'y1' in crop:
        ldict = {}
        exec(crop, globals(), ldict)
        x1 = ldict['x1']
        x2 = ldict['x2']
        y1 = ldict['y1']
        y2 = ldict['y2']
        # print(y1)
    else:
        raise Exception('I expect string to be like "y1=1003; y2=2248; x1=949; x2=1926"')
    return x1, x2, y1, y2

##
def annotate_simbad(img_file, fits_file, crop=None, save=True, fontScale=0.65, filter=None):
    # crop example: crop =  'y1=54; y2=3176; x1=2067; x2=7156'
    if save:
        from cv2 import putText, FONT_HERSHEY_SIMPLEX, LINE_AA, getTextSize
    header = fits.open(fits_file)[1].header
    if img_file is None:
        img = fits.open(fits_file)[1].data
        img = level_adjust(img)
        img_file = fits_file.replace('.fits', '.png')
        img[np.isnan(img)] = 0
        plt.imsave(img_file, img, origin='lower')
    img = plt.imread(img_file)
    if img.shape[2] == 4:
        img = img[..., :3]
    if not save:
        img = img[::-1, ...]
    # TODO: add pix to df and save csv
    wcs = WCS(header)
    print('querying SIMBAD')
    my_simbad = Simbad()
    my_simbad.add_votable_fields('otype')
    result_table = my_simbad.query_region(
        SkyCoord(ra=header['CRVAL1'],
                 dec=header['CRVAL2'],
                 unit=(u.deg, u.deg), frame='fk5'),
        radius=0.1 * u.deg)
    result_table = result_table.to_pandas()
    if filter:
        result_table = result_table[result_table['MAIN_ID'].str.contains(filter)]
        result_table.reset_index(drop=True, inplace=True)
    print(f'got {len(result_table)} results, converting to pixels')
    if len(result_table) == 0:
        raise Exception('no results')
    color = (100, 255, 255)
    result_table['color'] = [color]*len(result_table)
    result_table['OTYPE'].str.replace('Star', '*')
    icat = np.where(result_table['OTYPE'].str.contains('\*'))[0]
    for iicat in icat:
        result_table.at[iicat, 'color'] = (255, 255, 255)
    icat = np.where(result_table['OTYPE'].str.contains('ebula'))[0]
    for iicat in icat:
        result_table.at[iicat, 'color'] = (255, 100, 100)
    pix = np.zeros((len(result_table), 2))
    for ii in range(len(result_table)):
        ra = add_time(result_table['RA'][ii], 'h')
        dec = add_time(result_table['DEC'][ii], 'd')
        c = SkyCoord(ra=ra, dec=dec).to_pixel(wcs)
        pix[ii, :] = [c[0], c[1]]
    y1 = 0
    y2 = header['NAXIS2']
    x1 = 0
    x2 = header['NAXIS1']
    if crop is None:
        if img.shape[:2] != (y2, x2):
            raise Exception('image and hdu not the same size')
    else:
        x1, x2, y1, y2 = crop_xy(crop)
        if img.shape[:2] != (y2-y1, x2-x1):
            raise Exception('image is expected to fit crop size')
    inframe = (pix[:, 0] > x1) & (pix[:, 1] > y1) & (pix[:, 0] <= x2) & (pix[:, 1] <= y2)

    if np.sum(inframe) == 0:
        print(result_table)
        raise Exception('no results in frame')
    else:
        print(f'{np.sum(inframe)} results in frame')
    if save:
        # color = (155, 255, 255)
        thickness = 2
        if np.nanmax(img) <= 1:
            img = 255*img
            img = img.astype('uint8')
        for idx in np.where(inframe)[0]:
            txt = result_table['MAIN_ID'][idx]
            midhight = int(np.floor(getTextSize(txt,
                                            fontFace=FONT_HERSHEY_SIMPLEX,
                                            fontScale=fontScale,
                                            thickness=thickness)[0][1]/2))
            org = (int(np.round(pix[idx, 0] - x1)),
                   int(np.round(y2)) - int(np.round(pix[idx, 1])) + midhight)
            img = putText(img, txt, org,
                          FONT_HERSHEY_SIMPLEX, fontScale,
                          result_table['color'][idx], thickness, LINE_AA)
        plt.imsave(img_file.replace('.', '_ann.'), img)
    else:
        plt.figure()
        plt.imshow(img, origin='lower')
        for idx in np.where(inframe)[0]:
            plt.text(pix[idx, 0]-x1, pix[idx, 1]-y1,
                     result_table.iloc[idx]['MAIN_ID'], color=np.array(color)/255)
    return result_table[inframe]
##

def whiten_image(img):
    ''' prevent blue hue, rise red to min([green, blue]) '''
    img[..., 0] = np.max([img[..., 0], np.min(img[..., 1:], 2)], 0)
    return img

def reduce_color(img, bad=1, replace=np.min, thr='max', thratio=None):
    okay = [0, 1, 2]
    okay.pop(bad)
    good = replace(img[..., okay], 2)
    mask = np.ones(good.shape, bool)
    if thr is not None:
        if type(thr) == str:
            if thr == 'max':
                mask[good > img[..., bad]] = False
            else:
                raise Exception('noly max is allowed for string threshold')
    else:
        mask[good < thr] = False
    if thratio:
        rat = good/img[..., bad]
        mask[img[..., bad]/good < thratio] = False
    img[..., bad][mask] = good[mask]
    return img

def grey_zeros(img, bad=[0, 1, 2], thr=0, replace=np.min):
    if type(bad) == int:
        bad = [bad]
    for bd in bad:
        okay = [0, 1, 2]
        okay.pop(bd)
        layer = img[..., bd]
        mask = layer <= thr
        layer[mask] = replace(img[..., okay], 2)[mask]
        img[..., bd] = layer
    return img

def auto_plot(folder='ngc1672', exp='*_i2d.fits', method='rrgggbb', pow=[1, 1, 1], pkl=False, png=None, resize=False,
              plot=False, adj_args={'factor': 2}, fill=False, fill_func='max', smooth=False, max_color=False, opvar='rgb', core=False,
              crop=False, deband=False, deband_flip=False, blc=None, whiten=None, annotate=False, decimate=False, func=None, bar=False):
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
    png: bool | str | None
        False - don't save png. True - save png according to folder name. str - specify png name to save.
    resize: bool
        try resize to fit a 1920 by 1080 image. no cropping or aspect artio chabges. meant to reduce RAM and time.
    core: bool
        True to try focus on the core of the galaxy, in order to stretch the colors differently.
    adj_args: dict
        arguments to pass to level_adjust
    opvar: str
        what variable to return. 'rgb' or 'layers'
    deband: bool | int | list | ndarray
        remove banding noise 1/f. True or 1 for all layers, 2 or 'nircam' for nircam layers, False default. lots of time!
    deband_flip: bool | None | list
        True is intended for removing vertical stripes from MIRI data. None assigns True for filenames containing miri.
    blc: bool | None
        subtract non-zero minimum (baseline correction) and divide by maximum
    whiten: None | bool
        prevent blue hue, rise red to min([green, blue]). This is designed to make NIRCam more white than blue.
        by default whiten is True when at least one image is NIRCam
    Returns
    -------
    rgb: np.ndarray

    '''
    # TODO: clean small holes fast without conv, remove red background
    for search in ['./',
                   '../',
                   '/media/innereye/KINGSTON/JWST/data/',
                   '/home/innereye/astro/data/',
                   '/home/innereye/JWST/']:
        if os.path.isdir(search+folder):
            print(f"cd {search}")
            os.chdir(search)
            break
    if not os.path.isdir(folder):
        raise Exception('cannot find '+folder)
    from_log = False
    if type(exp) == pd.DataFrame:
        log = exp
        from_log=True
        path = np.array(log['file'])
        os.chdir(folder)
    elif type(exp) == str:
        if exp[:3] == 'log':
            from_log = True
            if len(exp) > 3:
                logpath = os.environ['HOME'] + '/astro/logs/' + exp.replace('log','')
                log = pd.read_csv(logpath)
            else:
                logpath = glob(os.environ['HOME'] + '/astro/logs/' + folder + '*')
                if len(logpath) == 1:
                    log = pd.read_csv(logpath[0])
                elif len(logpath) == 0:
                    print(logpath)
                    raise Exception('expected one log file or more')
                else:
                    log = pd.read_csv(logpath[0])
                    for ilog in range(1, len(logpath)):
                        log_next = pd.read_csv(logpath[ilog])
                        log = pd.concat([log, log_next])
                    log = log.reset_index()
            path = list(log['file'][log['chosen']])
            os.chdir(folder)
        else:
            path = list_files(os.getcwd()+'/'+folder, exp)
    else:
        path = exp
        os.chdir(folder)
    if len(path) == 0:
        raise Exception(f"no files in {os.getcwd()}")
    filt = filt_num(path)
    order = np.argsort(filt)
    path = np.asarray(path)[order]
    filt = filt[order]

    def make_rgb(prc=0):
        rgb = np.zeros((layers.shape[0], layers.shape[1], 3), float)
        for ll in range(3):
            if max_color == -1:
                lay = np.min(layers[:, :, iii[ll]], axis=2)
            elif max_color:  # convert zeros to nans
                lay = np.max(layers[:, :, iii[ll]], axis=2)
            else:
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

    def crval_fix(hd):
        if from_log and 'CRVAL1fix' in log.columns:
            logrow = np.where(log['file'] == path[ii])[0]
            if len(logrow) == 1:
                for cr in [1, 2]:
                    correct = log.iloc[logrow][f'CRVAL{cr}fix'].to_numpy()[0]
                    if ~np.isnan(correct):
                        hd[1].header[f'CRVAL{cr}'] = correct
        return hd

    todeband = np.zeros(len(path), bool)
    pkl_name = folder + '.pkl'
    if type(pkl) == str:
        pkl_name = pkl
        pkl = True
    if os.path.isfile(pkl_name) and pkl:
        if deband:
            raise Exception(f'deband BEFORE saving pickle. use deband=False or pkl=False or delete {pkl_name}')
        layers = np.load(pkl_name, allow_pickle=True)
        if len(path) < layers.shape[2]:
            os.chdir(search)
            path_full = list_files(os.getcwd()+'/'+folder, '*_i2d.fits')
            if len(path_full) == layers.shape[2]:
                filt_full = filt_num(path_full)
                order_full = np.argsort(filt_full)
                path_full = np.asarray(path_full)[order_full]
                include = [int(np.where(path_full == x)[0][0]) for x in path]
                layers = layers[:, :, include]
            else:
                raise Exception('which layer is which file?')
        if resize:
            wh = resize_wh(layers.shape[:2])
            layers = transform.resize(layers, wh)
    else:
        if deband:
            dbargs = {'func': np.percentile, 'prct': 10, 'flip': deband_flip}
            dbstr = ''
            if (type(deband) == list) or (type(deband) == np.ndarray):
                todeband = deband
            elif type(deband) == int or type(deband) == bool:
                todeband = np.ones(len(path))
                if type(deband) == int:
                    if deband == 50:
                        dbargs['func'] = np.nanmedian
                    else:
                        dbargs['func'] = np.nanpercentile
            elif deband == 'nircam':
                dbstr = ' nircam'
                todeband = np.array(['nircam' in x for x in path])
            elif deband == 'miri':
                todeband = np.array(['miri' in x for x in path])
            elif deband == 'n':
                dbstr = ' n'
                todeband = np.array(['n_i2d.' in x for x in path])
                dbargs['func'] = np.nanpercentile
            if deband_flip is None:
                deband_flip = np.array(['miri' in x for x in path])
                print('deband_flip')
                print(deband_flip)
            elif type(deband_flip) == bool:
                deband_flip = [deband_flip] * len(path)
            else:
                raise Exception('deband flip unresolved')

        for ii in range(len(path)):
            if ii == 0:
                hdu0 = fits.open(path[ii])
                hdu0 = crval_fix(hdu0)
                if decimate:
                    hdu0[1].data = hdu0[1].data[::decimate, ::decimate]
                    print('decimated')
                img = hdu0[1].data
                if fill:
                    img = hole_func_fill(img, func=fill_func)
                if resize:# make rescale size for wallpaper 1920 x 1080
                    wh = resize_wh(img.shape)
                    img = transform.resize(img, wh)
                if todeband[ii]:
                    dbargs['flip'] = deband_flip[ii]
                    print('going to deband'+dbstr+str(dbargs))
                    img = deband_layer(img, **dbargs)
                    print('done deband 0')
                layers = np.zeros((img.shape[0], img.shape[1], len(path)))
                hdr0 = hdu0[1].header
                hdu0.close()
            else:
                hdu = fits.open(path[ii])
                if decimate:
                    hdu[1].data = hdu[1].data[::decimate, ::decimate]
                if fill:
                    hdu[1].data = hole_func_fill(hdu[1].data,  func=fill_func)
                hdu = crval_fix(hdu)
                if todeband[ii]:
                    dbargs['flip'] = deband_flip[ii]
                    hdu[1].data = deband_layer(hdu[1].data, **dbargs)
                    print(f'done deband {ii}')
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
            x1, x2, y1, y2 = crop_xy(crop)
        else:
            lay3 = [0, int(layers.shape[2]/2), layers.shape[2]-1]
            imtochoose = layers[..., lay3].copy()
            for i3 in range(3):
                imtochoose[..., i3] = level_adjust(imtochoose[..., i3])
            plt.figure()
            plt.imshow(level_adjust(imtochoose), origin='lower')
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
            plt.show(block=True)
            p1, p2 = click_coordinates
            x1, y1 = int(min(p1[0], p2[0])), int(min(p1[1], p2[1]))
            x2, y2 = int(max(p1[0], p2[0])), int(max(p1[1], p2[1]))
            crop = f'y1={y1}; y2={y2}; x1={x1}; x2={x2}'
            print(crop)
        layers = layers[y1:y2, x1:x2]
        core_str = '_crop'
    else:
        core_str = ''
    iii = None
    empty = np.zeros(len(path), bool)
    for lay in range(layers.shape[2]):
        if np.mean(np.isnan(layers[:, :, lay])) == 1 or layers[:, :, lay].sum() == 0:
            empty[lay] = True
        else:
            if func:
                layers[:, :, lay] = level_adjust(func(layers[:, :, lay]), **adj_args)
            else:
                layers[:, :, lay] = level_adjust(layers[:, :, lay], **adj_args)
            # if fill:
            #     xy = hole_xy(layers[:, :, lay])
            #     size = hole_size(layers[:, :, lay], xy, plot=False)
            #     layers[:, :, lay] = hole_disk_fill(layers[:, :, lay], xy, size, larger_than=0, allowed=0.33)
            if smooth:
                layers[:, :, lay] = smooth_yx(layers[:, :, lay], 5, 2)
    # combine colors by method
    layers = layers[..., ~empty]
    if bar:
        if type(bar) == bool:
            bar_square_size = int(layers.shape[0]*0.2/layers.shape[2])
        else:
            bar_square_size = int(bar)
        for ilay in range(layers.shape[2]):
            layers[bar_square_size * ilay: bar_square_size * ilay + bar_square_size, :bar_square_size, :] = 0
            layers[bar_square_size * ilay: bar_square_size * ilay + bar_square_size, :bar_square_size, ilay] = 1
    path = path[~empty]
    rgb = None
    ismiri = ['miri' in x for x in path]
    if layers.shape[2] == 2:
        ir = [1]
        ib = [0]
        ig = [0, 1]
        iii = [ir, ig, ib]
    elif method == 'rrgggbb':
        ncol = np.floor(layers.shape[-1] / 3)
        ib = np.arange(0, ncol).astype(int)
        ir = np.arange(layers.shape[-1] - ncol, layers.shape[-1]).astype(int)
        ig = np.arange(ib[-1] + 1, ir[0]).astype(int)
        iii = [ir, ig, ib]
    elif method[:3] == 'mnn':  # Miri Nircam Nircam, could also be mnnw for whiteish
        ir = np.where(ismiri)[0]
        inircam = np.where(~np.asarray(ismiri))[0]
        nb = np.ceil(len(inircam)/2)
        ib = np.arange(0, nb).astype(int)
        ig = np.arange(ib[-1] + 1, ir[0]).astype(int)
        iii = [ir, ig, ib]
    elif method == 'mtn':  # Miri Total NirCam
        ir = np.where(ismiri)[0]
        ib = np.where(~np.asarray(ismiri))[0]
        ig = np.arange(layers.shape[2]).astype(int)
        iii = [ir, ig, ib]
    elif method == 'filt05':
        rgb = assign_colors_by_filt(layers, filt)
        rgb = rgb*255
        rgb = rgb.astype('uint8')
        blc = False
    elif method[:4] == 'filt':
        col = matplotlib.cm.jet(filt / np.max(filt))[:, :3]  # [:, ::-1]
        rgb = assign_colors(layers, col)
        for ic in range(3):
            rgb[:, :, ic] = rgb[:, :, ic] ** pow[ic] * 255
        rgb = rgb.astype('uint8')
    if rgb is None:
        rgb = make_rgb()
    if whiten is None:
        if all(ismiri):
            whiten = False
        else:
            whiten = True
    if blc is None:
        if method == 'filt':
            blc = True
        else:
            blc = False
    if blc:
        rgb = blc_image(rgb)
    if whiten:
        rgb = whiten_image(rgb)
    if bar:
        colorbar = rgb[: bar_square_size * ilay + bar_square_size, :bar_square_size, :]
        cbmax = np.max(colorbar)
        print(cbmax)
        colorbar = colorbar.copy()/cbmax*255  # ).astype('uint8')
        colorbar = colorbar.astype('uint8')
        rgb[: bar_square_size * ilay + bar_square_size, :bar_square_size, :] = colorbar
    if plot:
        plt.figure()
        plt.imshow(rgb, origin='lower')
        plt.show()
        if annotate:
            print('no annotate for plot, only works for save')
    if png is None:
        png = f"{method}_fac{adj_args['factor']}.jpg"
    if png:
        if type(png) == str:
            png_name = png
        else:
            png_name = folder+core_str+'.png'
        if png_name[-3:] == 'jpg':
            plt.imsave(png_name, rgb, origin='lower', pil_kwargs={'quality':95})
        else:
            plt.imsave(png_name, rgb, origin='lower')
        if annotate:
            if not crop:
                crop = None
            elif type(crop) != str:
                raise Exception(f'what is crop? crop = {crop}')
            if type(annotate) == bool:
                fontScale = 0.6
            else:
                fontScale = annotate
            annotate_simbad(png_name, path[0], crop=crop, save=True, fontScale=fontScale)
    return eval(opvar)


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


def assign_colors_by_filt(layers, filt, legend=True, subtract_blue=0.5, colormap='hsv', udflip=True, blc=True):
    """
    Assign color according to frequency.

    Parameters
    ----------
    layers : 3D ndarray
        Adjusted data between 0 and 1
    filt : list | ndarray
        A vector of filter numbers
    legend : bool, optional
        True for making a legend at the bottom
    subtract_blue : float | None, optional
        Subtract the blue color before considering frequencies. 0.5 would 
        subtract half of blue. 0 = no action, None to subtract 50 (visible blue)

    Returns
    -------
    rgb : ndarray
        0 to 1 color image.

    """
    # col = matplotlib.cm.jet(filt / np.max(filt))[:, :3]
    if udflip:
        layers=layers[::-1, ...]
    if subtract_blue is None:
        filt = filt - 50
        subtract_blue = 0
    filtnorm = filt-min(filt)*subtract_blue
    filtnorm = filtnorm/max(filtnorm)
    if colormap == 'jet':
        col = matplotlib.cm.jet(filtnorm)[:, :3]
    elif colormap == 'rainbow':
        col = matplotlib.cm.rainbow(filtnorm)[:, :3]
    elif colormap == 'hsv':
        filtnorm = (1-filtnorm)*0.680
        col = matplotlib.cm.hsv(filtnorm)[:, :3]
    if legend:
        from cv2 import putText, FONT_HERSHEY_SIMPLEX, LINE_AA, getTextSize
        square_length = int(layers.shape[0]/10/layers.shape[2])
        for ilayer in range(layers.shape[2]):
            start = square_length*ilayer + layers.shape[0] - layers.shape[2] * square_length
            end = start + square_length
            layers[start:end, :square_length, :] = 0
            layers[start:end, :square_length, ilayer] = 1
            txt = str(int(filt[ilayer])).zfill(3)
            org = (int(square_length), int(start+square_length+2))
            thickness = int(square_length/8)
            ((fw,fh), baseline) = getTextSize(
                "", fontFace=FONT_HERSHEY_SIMPLEX, fontScale=100, thickness=thickness) # empty string is good enough
            factor = (fh-1) / 100
            height_in_pixels = square_length - 4 # or 20, code works either way
            fontScale = (height_in_pixels - thickness) / factor
            for jlayer in range(layers.shape[2]):
                layer = layers[..., jlayer]
                layer = putText(layer*255, txt, org,
                          FONT_HERSHEY_SIMPLEX, fontScale, 255, thickness, LINE_AA)
                layers[..., jlayer] = layer/255
    rgb = assign_colors(layers, col)
    if blc:
        rgb = blc_image(rgb)
    if udflip:
        rgb = rgb[::-1, ...]
    return rgb


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


def log(arr, small=0.01):
    arr[arr <= 0] = small
    arr = np.log10(arr)
    return arr


drive = '/media/innereye/KINGSTON/JWST/'
def download_by_log(log_csv, tgt=None, overwrite=False, wget=False, path2data=None):
    if log_csv[0] != '/':
        log_csv = '/home/innereye/astro/logs/' + log_csv
    if path2data is None:
        path2data = drive
    if os.path.isdir(path2data):
        os.chdir(path2data)
    else:
        raise Exception('where is the drive?')
    if tgt is None:
        tgt = log_csv.split('_')[0].split('/')[-1]
    if not os.path.isdir('data/' + tgt):
        os.system('mkdir data/' + tgt)
    if os.path.isfile(log_csv):
        chosen_df = pd.read_csv(log_csv)
        files = list(chosen_df['file'][chosen_df['chosen']])
        print(f'downloading {tgt} by log')
        download_fits_files(files, destination_folder='data/' + tgt, overwrite=overwrite, wget=wget)
    else:
        print('where is the log file?')


def resize_with_padding(img, target_size=(1200, 675)):
    """
    Resize an image to the target size, maintaining aspect ratio by padding the
    remaining area.
    """
    padding_color = 0
    # img = plt.imread(image_path)
    # Determine the target aspect ratio
    target_aspect = target_size[0] / target_size[1]
    img_aspect = img.shape[1] / img.shape[0]
    # Resize the image to fit within the target size, maintaining aspect ratio
    if img_aspect > target_aspect:
        # Image is wider than target aspect ratio
        new_width = target_size[0]
        new_height = round(new_width / img_aspect)
    else:
        # Image is taller than target aspect ratio
        new_height = target_size[1]
        new_width = round(new_height * img_aspect)
    # Resize the image
    resized_img = transform.resize(img, (new_height, new_width), anti_aliasing=True)
    # Create a new image with the target size and fill with the padding color
    padded_img = np.zeros((target_size[1], target_size[0])) * padding_color
    # Calculate the position to paste the resized image onto the background
    y_offset = (target_size[1] - new_height) // 2
    x_offset = (target_size[0] - new_width) // 2
    # Place the resized image in the center of the padded background
    padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
    return padded_img

def resize_to_under_2mb(image: np.ndarray, max_size_mb: float = 2.0, min_scale: float = 0.1, step: float = 0.9) -> np.ndarray:
    """
    Resize a 2D or 3D image using skimage.transform to ensure it is under a given size when saved as JPEG.

    Args:
        image (np.ndarray): The input image (2D grayscale or 3D RGB).
        max_size_mb (float): Max file size in MB. Default is 2MB.
        min_scale (float): Minimum scaling factor allowed.
        step (float): Scale reduction factor per iteration (e.g., 0.9 = reduce by 10% per step).

    Returns:
        np.ndarray: Resized image as a uint8 array.
    """
    scale = 1.0
    max_bytes = max_size_mb * 1024 * 1024
    while scale >= min_scale:
        # Compute new shape
        new_shape = (np.array(image.shape[:2]) * scale).astype(int)

        # Resize with skimage
        if image.ndim == 3:
            resized = transform.resize(image, (*new_shape, image.shape[2]), anti_aliasing=True)
        else:
            resized = transform.resize(image, new_shape, anti_aliasing=True)

        resized_uint8 = img_as_ubyte(resized)

        # Save to a temp file using plt to check size
        if len(image.shape) == 2:
            plt.imsave('tmprs.jpg', resized_uint8, pil_kwargs={'quality': 95}, cmap='gray')
        else:
            plt.imsave('tmprs.jpg', resized_uint8, pil_kwargs={'quality': 95})
        size = os.path.getsize('tmprs.jpg')
        # os.remove(tmp_file.name)
        # print(f"{new_shape}")
        if size <= max_bytes:
            return resized_uint8
        scale *= step
    raise ValueError("Could not resize image to be under 2MB within the given scale range.")



def resize_to_under_1mp(image: np.ndarray, max_pix: int=1000000) -> np.ndarray:
    """
    Resizes an image so that its total number of pixels is less than 1,000,000,
    preserving the aspect ratio. Uses skimage for resizing.
    
    Parameters:
        image (np.ndarray): Input image as a NumPy array.
    
    Returns:
        np.ndarray: Resized image as a NumPy array.
    """
    h, w = image.shape[:2]
    total_pixels = h * w
    if total_pixels <= max_pix:
        return image  # No resizing needed
    scale = (1_000_000 / total_pixels) ** 0.5
    new_h = int(h * scale)
    new_w = int(w * scale)
    resized = transform.resize(image, (new_h, new_w), anti_aliasing=True)
    # Convert to uint8 if original image was in uint8
    if image.dtype == np.uint8:
        resized = img_as_ubyte(resized)
    return resized



def overlap(files):
    """Check if images overlap, chat GPT, not very good"""
    def get_corners(wcs, shape):
        ny, nx = shape[-2], shape[-1]
        pix_coords = np.array([[0, 0], [0, ny], [nx, 0], [nx, ny]])
        world_coords = wcs.pixel_to_world(pix_coords[:, 0], pix_coords[:, 1])
    
        # If WCS is celestial + extra (e.g., spectral), extract just celestial
        if hasattr(world_coords, 'ra') and hasattr(world_coords, 'dec'):
            pass  # Already a SkyCoord or similar object
        elif isinstance(world_coords, tuple):
            world_coords = world_coords[0]  # Assumes 1st element is celestial
        else:
            raise ValueError("Could not extract celestial coordinates from WCS output.")
        corners = {'ra_min': world_coords.ra.min(),
                   'ra_max': world_coords.ra.max(),
                   'dec_min': world_coords.dec.min(),
                   'dec_max': world_coords.dec.max()}
        return corners
    def area(c):
        return (c['ra_max'] - c['ra_min']) * (c['dec_max'] - c['dec_min'])

    if not os.path.isfile(files[0]):
        if '/' in files[0]:
            files = [f.split('/')[-1] for f in files]
        files = [mast_url+f for f in files]
    # get headers to evaluate overlap (could be faster f=with fitsheader)
    headers = []
    for file in files:
        with fits.open(file, use_fsspec=True) as hdul:
            headers.append(hdul[1].header)
            # shapes.append(hdul[1].shape)
    shapes = []
    for header in headers:
        naxis = header['NAXIS']
        shapes.append(tuple(header[f'NAXIS{i}'] for i in range(naxis, 0, -1)))
    corners = []
    for iheader in range(len(headers)):
        wcs = WCS(headers[iheader])
        shape = shapes[iheader]
        corners.append(get_corners(wcs, shape))
    percent_overlap = np.zeros((len(headers), len(headers)))
    for ii in range(len(headers)):
        for jj in range(len(headers)):
            corners1 = corners[ii]
            corners2 = corners[jj]
            if area(corners1) <= area(corners2):
                small, large = corners1, corners2
            else:
                large, small = corners1, corners2
            ra_overlap_min = max(small['ra_min'], large['ra_min'])
            ra_overlap_max = min(small['ra_max'], large['ra_max'])
            dec_overlap_min = max(small['dec_min'], large['dec_min'])
            dec_overlap_max = min(small['dec_max'], large['dec_max'])
            # Check if there's any overlap at all
            if ra_overlap_max <= ra_overlap_min or dec_overlap_max <= dec_overlap_min:
                overlap_fraction = 0.0
            else:
                overlap_area = (ra_overlap_max - ra_overlap_min) * (dec_overlap_max - dec_overlap_min)
                small_area = area(small)
                overlap_fraction = overlap_area / small_area
            percent_overlap[ii, jj] = overlap_fraction
    threshold = 0.5
    # Build graph
    G = nx.Graph()
    n = percent_overlap.shape[0]
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if percent_overlap[i, j] > threshold:
                G.add_edge(i, j)
    # Get connected components (groups)
    groups = list(nx.connected_components(G))
    groups = [list(g) for g in groups]
    # Print groups
    return groups, percent_overlap

def cluster_coordinates(coords, threshold=0.001):
    coords = np.array(coords)
    n = len(coords)
    visited = np.zeros(n, dtype=bool)
    labels = np.full(n, -1, dtype=int)  # -1 for unassigned
    cluster_id = 0
    def is_close(p1, p2):
        return abs(p1[0] - p2[0]) <= threshold and abs(p1[1] - p2[1]) <= threshold
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        labels[i] = cluster_id
        queue = [i]
        while queue:
            current = queue.pop()
            for j in range(n):
                if not visited[j] and is_close(coords[current], coords[j]):
                    visited[j] = True
                    labels[j] = cluster_id
                    queue.append(j)
        cluster_id += 1
    return labels

if __name__ == '__main__':
    auto_plot('NGC2506G31', exp = '*fits', png='rgb4log_deband.jpg',adj_args={'factor':4},
          func=log, method='rrgggbb', fill=False, pkl=False, deband=10, deband_flip=True)

