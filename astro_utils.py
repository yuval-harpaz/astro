
from astropy.io import fits
from astropy import wcs
from reproject import reproject_interp
import os
import glob
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from astropy.convolution import Ring2DKernel
from scipy.ndimage.filters import median_filter
from astroquery.mast import Observations

def list_files(parent, search='*_cal.fits', exclude='', include=''):
    """
    List all fits files in subfolders of path.
    """
    os.chdir(list_files.__code__.co_filename[:-14]+'data')
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
    ring = Ring2DKernel(9, 3)
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


if __name__ == '__main__':
    include = 'jw02732-c1001_t004_miri_ch2'
    download_fits('ngc 7319', extension='.fits', mrp=True, include=include, ptype='')

    # path3 = list_files('/home/innereye/JWST/Quintet/ngc_7319/MAST_2022-08-27T0857/JWST/', search='*s3d.fits')
    # xy, size = mosaic_xy(path3, plot=True)
    # manifest = download_fits('IC 1623B')
    # path = list_files('ngc_628', search='*nircam*.fits')
    # # get filename from full path
    # layers = reproject(path, project_to=1)
    #
    # # manifest = download_fits('ngc 628', include=['_miri_', '_nircam_', 'clear'])
    # # path = list_files('data/ngc_628', search='*miri*.fits')
    # # path = list_files('Cartwheel/long', search='*.fits')
    # path = list_files('ngc_628', search='*miri*1000*.fits')
    # median = mosaic(path, plot=True, method='median')
    # mn = 0.11
    # mx = 1.7
    # plt.clim(mn, mx)
    # plt.show(block=False)
    # img = (median-mn)/(mx-mn)*255
    # img[img < 0] = 0
    # img[img > 255] = 255
    # img = img.astype(np.uint8)
    # img[1947:1952, 1019:1024] = 255
    # img[775:777, 2042:2045] = 255
    # plt.imsave('median.png', np.flipud(img), cmap='gray')
    # plt.imsave('median_hot.png', np.flipud(img), cmap='hot')
    # layers = mosaic(path, plot=False, method='layers')
    # layers[layers == 0] = np.nan
    # rgb = np.zeros((median.shape[0], median.shape[1], 3))
    # for ii in range(3):
    #     layer1 = np.nanmedian(layers[:, :, ii*8:ii*8+8], axis=2)
    #     layer1[np.isnan(layer1)] = np.nanmedian(layer1)
    #     rgb[:, :, ii] = layer1
    # img = (rgb - mn) / (mx - mn) * 255
    # img[img < 0] = 0
    # img[img > 255] = 255
    # img = img.astype(np.uint8)
    # img[1945:1952, 1018:1025] = 255
    # img[773:777, 2042:2046] = 255
    # plt.imsave('median_rgb.png', img)
    # mosaic(path, plot=True, method='mean')
    # plt.clim(0.2, 1.5)
    # plt.show(block=False)


    # path = list_files('/home/innereye/JWST/MAST_2022-08-09T0239/JWST/', include='_02101_')
    # canvas = mosaic(path, plot=True, clip=[7.45, 10], method='layers')
    # canvas = mosaic(path, clip=[7.45, 10], plot=True)
    # exclude = ['jw02727002001_02105_00001_nrcb2', 'cal/', 'Link ']
    # parent = '/home/innereye/JWST/MAST_2022-08-09T0229/JWST'
    # path = list_files(parent, exclude=exclude)
    # bestx, besty, canvaso = optimize_xy(canvas, square_size=100, tests=9, plot=True)
    # # bestx = [0, 2, 1, 1]
    # # besty = [0, -1, -1, 1]
    # x1 = 550
    # y1 = 500
    # plt.figure()
    # for ii in range(canvas.shape[2]):
    #     plt.subplot(2,5, ii+1)
    #     plt.imshow(canvas[x1:x1+100,y1:y1+100,ii], cmap='hot')
    #     plt.axis('equal')
    #     plt.clim(7.45, 10)
    #     plt.axis('off')
    #     plt.show(block=False)
    # plt.subplot(2, 5, 5)
    # plt.imshow(np.nanmean(canvas[x1:x1+100,y1:y1+100, :],axis=2), cmap='hot')
    # plt.axis('equal')
    # plt.clim(7.45, 10)
    # plt.axis('off')
    # plt.show(block=False)
    # for ii in range(canvaso.shape[2]):
    #     plt.subplot(2,5, ii+6)
    #     plt.imshow(canvaso[x1:x1+100,y1:y1+100,ii], cmap='hot')
    #     plt.axis('equal')
    #     plt.clim(7.45, 10)
    #     plt.axis('off')
    #     plt.show(block=False)
    # plt.subplot(2, 5, 10)
    # plt.imshow(np.nanmean(canvaso[x1:x1+100,y1:y1+100, :],axis=2), cmap='hot')
    # plt.axis('equal')
    # plt.clim(7.45, 10)
    # plt.axis('off')
    # plt.show(block=False)
    #
    # xy, size = mosaic_xy(path)
    # data = np.empty((size[0, 0], size[0, 1], size.shape[0]))
    # for ii, p in enumerate(path):
    #     hdul = fits.open(p)
    #     data[:, :, ii] = hdul[1].data.copy()
    # plt.figure()
    # plt.imshow(np.mean(data,axis=2), cmap='hot')
    # # plt.imshow(data[:,:,0], cmap='hot')
    # plt.axis('equal')
    # plt.clim(7.45, 10)
    # # plt.xlim(0, x1)
    # # plt.ylim(0, y1)
    # plt.axis('off')
    # plt.show(block=False)
    # mask = np.zeros(data.shape[:2], dtype=bool)
    # mask[np.sum(data > 9, axis=2) == data.shape[2]] = True
    # mask[np.sum(data < 2.5, axis=2) == data.shape[2]] = True
    # mask[:12, :] = True
    # mask[-12:, :] = True
    # mask[:, :12] = True
    # mask[:, -12:] = True
    # data[mask, ...] = np.nan
    # avg = np.nanmean(data, axis=2)
    # avg[mask] = 0
    # plt.figure()
    # plt.imshow(avg, cmap='hot')
    # plt.axis('equal')
    # plt.clim(7.45, 10)
    # plt.axis('off')
    # plt.show(block=False)
    #
    # mosaic(path, plot=False, clip=[0.35, 0.6])
    # canvas = mosaic(path, plot=True, clip=[0.35, 0.6])
    # parent = '/home/innereye/JWST/MAST_2022-08-09T0229/JWST/jw02727002001_02105_00001_nrcb2'
    # path = list_files(parent)
    # mosaic_xy(path, plot=True)


