# https://raw.githubusercontent.com/Rachmanin0xFF/jwst-twitter-bot/main/grabber.py
# https://github.com/Rachmanin0xFF/jwst-twitter-bot/blob/main/image_fetcher.py
# @author Adam Lastowka


from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
import astroquery
from astroquery.mast import Observations
from astropy.time import Time
from astropy.table import vstack, Table


def to1(x):
    return 0.5 + (2.0 * x - 2.0) / (2 * np.sqrt((2.0 * x - 2.0) * (2.0 * x - 2.0) + 1.0))


def expand_highs(x):
    return np.piecewise(x, [x <= 0.9, x > 0.9],
                        [lambda x: x * 0.8 / 0.9, lambda x: 100.0 / 9.0 * (x - 0.9) ** 2 + 0.8 * x / 0.9])


def image_histogram_equalization(image, number_bins=10000, ignore0=False):
    # from http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    # get image histogram
    shape = image.shape
    image = image.flatten()
    if ignore0:
        include = image > 0
    else:
        include = np.ones(len(image), bool)
    image_histogram, bins = np.histogram(image[include], number_bins, density=True)
    cdf = image_histogram.cumsum()  # cumulative distribution function
    cdf = cdf / cdf[-1]  # normalize
    # use linear interpolation of cdf to find new pixel values

    image[include] = np.interp(image[include], bins[:-1], cdf)
    return image.reshape(shape)

def nanmask(arr):
    mask = np.isnan(arr)
    if np.sum(mask) > 0:
        arr[mask] = 0
    else:
        mask = None
    return arr, mask


def level_adjust(fits_arr, negative='consider', lims=[0.03, 0.98], factor=4.0, ignore0=False):
    """
    Stretch colors of an image. The image is normalized between zero and one, and then manipulated to show faint
    values. Based on https://github.com/Rachmanin0xFF/jwst-twitter-bot
    Args:
        fits_arr: 2D np.ndarray
            The image
        negative: str | anything
            'consider' allows negative colors when computing histogram. otherwise normalization is only for
            positive values.
        lims: list
            lower and upper lim for normalization. if >= one, quantiles are assumed. otherwise these are the
            actual values between which the image is normalized.
        factor: int
            you can play with it but I would leave it alone. you can try factor=1 to avoid power, sometimes nice.
        ignore0: bool
            ignore zeros (or values below lower limit) when using image_histogram_equalization

    Returns:
        adjusted image
    """
    fits_arr, mask = nanmask(fits_arr)
    hist_dat = fits_arr.flatten()
    if negative == 'consider':
        hist_dat = hist_dat[np.nonzero(hist_dat)]
    else:
        hist_dat = hist_dat[hist_dat > 0]
    nonzeros = np.abs(np.sign(fits_arr))
    if lims[1] > 1:
        minval, maxval = lims
    else:
        minval = np.quantile(hist_dat, lims[0])
        maxval = np.quantile(hist_dat, lims[1])
        # print([minval, maxval])
        # minval, maxval = np.quantile(hist_dat, lims)

    if lims == [0.03, 0.98]:  # for backward compatibility
        rescaled = (fits_arr - minval) / (maxval - minval)
        rescaled_no_outliers = np.maximum(rescaled, np.quantile(rescaled, 0.002))
        rescaled_no_outliers = np.minimum(rescaled_no_outliers, np.quantile(rescaled_no_outliers, 1.0 - 0.002))
    else:
        no_outliers = np.maximum(fits_arr, minval)
        no_outliers = np.minimum(no_outliers, maxval)
        rescaled_no_outliers = (no_outliers - minval) / (maxval - minval)
        rescaled = rescaled_no_outliers
    img_eqd = image_histogram_equalization(rescaled_no_outliers, ignore0=ignore0)
    img_eqd = (pow(img_eqd, factor) + pow(img_eqd, factor*2) + pow(img_eqd, factor*4)) / 3.0
    adjusted = expand_highs((img_eqd + to1(rescaled)) * 0.5)
    if mask is not None:
        adjusted[mask] = np.nan
    return np.clip(adjusted * nonzeros, 0.0, 1.0)

def get_JWST_products_from(start_time=None, end_time=None, release=True):
    """
    Gets calib=3 level public NIRCAM and MIRI products within a given time range.
    Also pushes any downloaded observations onto the obs_data dictionary.
    Parameters:
        start_time: A number representing the MJD (Modified Julian Date) start time of the range
        end_time: MJD end of time range
    Returns:
        An AstroPy Table containing a list of I2D products.
    """
    # Observations.query_criteria() output columns:
    # ['dataproduct_type', 'calib_level', 'obs_collection', 'obs_id', 'target_name',
    #  's_ra', 's_dec', 't_min', 't_max', 't_exptime', 'wavelength_region', 'filters',
    #  'em_min', 'em_max', 'target_classification', 'obs_title', 't_obs_release',
    #  'instrument_name', 'proposal_pi', 'proposal_id', 'proposal_type', 'project',
    #  'sequence_number', 'provenance_name', 's_region', 'jpegURL', 'dataURL',
    #  'dataRights', 'mtFlag', 'srcDen', 'intentType', 'obsid', 'objID']

    # Output table columns:
    # ['obsID', 'obs_collection', 'dataproduct_type', 'obs_id', 'description', 'type', 'dataURI', 'productType',
    #  'productGroupDescription', 'productSubGroupDescription', 'productDocumentationURL', 'project',
    #  'prvversion', 'proposal_id', 'productFilename', 'size', 'parent_obsid', 'dataRights', 'calib_level']
    obs_data = {}
    if end_time is None:
        end_time = Time.now().mjd
    if start_time is None:
        start_time = end_time - 1
    print("Querying MAST from " + Time(start_time,format='mjd').utc.iso + " to " + Time(end_time,format='mjd').utc.iso )
    # log_print("Q " + str(start_time) + " " + str(end_time))
    if release:
        obsByName = Observations.query_criteria(obs_collection="JWST",
                                                t_obs_release=[start_time, end_time],
                                                calib_level=3,
                                                dataproduct_type="image")
    else:
        obsByName = Observations.query_criteria(obs_collection="JWST",
                                                instrument_name=["NIRCAM", "MIRI", "NIRCAM/IMAGE", "MIRI/IMAGE", "NIRISS/IMAGE"],
                                                t_min=[start_time, end_time],
                                                calib_level=3,
                                                dataproduct_type="image")

    # TODO get image from obsByName['jpegURL'] using such url\
    #  https://mast.stsci.edu/portal/Download/file/JWST/product/jw02725-o072_t015_nircam_f212n-wlp8-nrca4_wfscmb-05.jpg
    all_result_count = len(obsByName)
    print(obsByName.colnames)
    obsByName = obsByName[(obsByName["dataRights"]=="PUBLIC")]
    print("Number of public results from JWST NIRCAM/MIRI: " + str(len(obsByName)))
    print("Number of exclusive/restricted results: " + str(all_result_count-len(obsByName)))
    print(obsByName[:4])
    for o in obsByName:
        obs_data[o["obs_id"]] = o
    alli2d = []
    k = 0
    for o in obsByName:
        k += 1
        print("Opening object " + str(k) + "/" + str(len(obsByName)))
        try:
            data_products = None
            while data_products is None:
                try:
                    data_products = Observations.get_product_list(o)
                except OSError:
                    data_products = None
            calibrated = data_products[(data_products['calib_level'] >= 3)]
            print("Total calib_level=3 products in objects:" + str(len(calibrated)))
            i2d = calibrated[(calibrated["productSubGroupDescription"] == "I2D")]
            print("Total calib_level=3 && I2D products in result:" + str(len(i2d)))
            alli2d.append(i2d)
        except astroquery.exceptions.InvalidQueryError:
            print("Invalid query!")
            pass
    if len(alli2d) == 0:
        return Table({})
    else:
        return vstack(alli2d)


if __name__ == '__main__':
    from astro_utils import *
    _ = auto_plot('NGC-1433', exp='log', png=False, pow=[1, 1, 1], pkl=True, method='mnn', resize=True, plot=True,
                  adj_args={'lims': [0.95, 1.0], 'ignore0': True})  # adj_args={'lims': [0.03, 1.0]}
    # fname = '/home/innereye/JWST/ngc1512/jw02107-o006_t006_miri_f770w_i2d.fits'
    # hdu = fits.open(fname)
    # raw = hdu[1].data
    # val_arr = level_adjust(hdu[1].data)
    # hdu.close()
    # plt.figure()
    # plt.imshow(val_arr)
