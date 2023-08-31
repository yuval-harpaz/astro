from astro_utils import *
auto_plot('IC348-MOSAIC', exp='log', png='deband.png', pkl=True, resize=False, method='rrgggbb', plot=False,
           max_color=False, fill=False, deband=True, adj_args={'factor': 2})
# auto_plot('HH46', exp='*.fits', png='deband.png', pow=[1, 1, 1], pkl=False, resize=True, method='rrgggbb', plot=False,
#           adj_args={'factor': 4}, max_color=False, fill=True, deband=True, )
# from time import time
#
# ##
# auto_plot('HH46', png='deband.png', pow=[1, 1, 1], pkl=True, resize=False, method='rrgggbb', plot=False,
#           adj_args={'factor': 2}, max_color=False, fill=True, deband=True)
# ##
#
#
# win = 101
# data = fits.open('/media/innereye/My Passport/Data/JWST/data/HH46/jw04441-o096_t008_nircam_clear-f335m_i2d.fits')[1].data.copy()
# data = level_adjust(data, factor=2)
# t0 = time()
# hp = data.copy()
# print('medfilt 0...')
# for ii in range(hp.shape[0]):
#     hp[ii, :] = medfilt(hp[ii, :], 101)
#     print(f'{ii}/{hp.shape[0]}', end='\r')
# t1 = time()-t0
#
# t0 = time()
# half0 = int(win/2)
# half1 = win-half0
# hp = data.copy()
# print('medfilt 0...')
# for ii in range(hp.shape[0]):
#     toavg = np.nan * np.ones((data.shape[1] + win - 1, win))
#     for shift in np.arange(win):
#         toavg[shift:data.shape[1]+shift, shift] = data[ii, :]
#     hp[ii, :] = np.nanmedian(toavg, axis=1)[half0:-half1+1]
#     print(f'{ii}/{hp.shape[0]}', end='\r')
# t2 = time()-t0
#
# t0 = time()
# smoothed = smooth_width(data, win=101)
# t3 = time()-t0
# t0 = time()
# smoothed = smooth_width(smoothed.T, win=101)
# t4 = time()-t0
#
# t0 = time()
# kernel = Gaussian2DKernel(12)
# lp = convolve(data, kernel=kernel)
# t5 = time()-t0
#
# # half0 = int(win/2)
# # half1 = win-half0
# # for ic in range(data.shape[1]):
# #     # create a matrix of nans with the same shape as the data
# #     toavg = np.nan * np.ones((data.shape[0]+win-1, win))
# #     for shift in np.arange(win):
# #         toavg[shift:data.shape[0]+shift, shift] = data[:, ic]
# #     if 'mean' in method:
# #         avg = np.nanmean(toavg, axis=1)
# #     elif 'median' in method:
# #         avg = np.nanmedian(toavg, axis=1)
# def deband_layer(layer):
#     # kernel = Ring2DKernel(50, 3)
#     # print('ring...')
#     # kernel = Gaussian2DKernel(12)
#     # print('gaus...')
#     # lp = convolve(layer, kernel=kernel)
#     # lp = median_filter(layer, footprint=kernel.array)
#     # lp = smooth_yx(layer, 101, 1)
#     hp = layer.copy()
#     print('medfilt 0...')
#     for ii in range(hp.shape[0]):
#         hp[ii, :] = medfilt(hp[ii, :], 101)
#         print(f'{ii}/{hp.shape[0]}', end='\r')
#     print('medfilt 1...')
#     lp = layer.copy()
#     for ii in range(hp.shape[0]):
#         lp[:, ii] = medfilt(hp[:, ii], 101)
#         # print(f'{ii}/{hp.shape[0]}')
#     hp = layer - hp
#     clean = lp + hp
#     clean[clean < 0] = 0
#     return clean
# # _ = auto_plot('NGC-1433', exp='log', png=False, pow=[1, 1, 1], pkl=True, method='mnn', resize=True, plot=True, adj_args={'lims': [0.95, 1.0], 'ignore0': True})  # adj_args={'lims': [0.03, 1.0]}
# # from astro_list_ngc import
# # auto_plot('NGC-346', exp='*w_i2d.fits', png='all.png', pow=[1, 1, 1], pkl=True, method='mnn', crop=False, plot=False)
# # auto_plot('ORIBAR-IMAGING-NIRCAM', exp='*_f*.fits', png='f.png', pow=[1, 1, 1], pkl=False, crop=True, method='rrgggbb')
# # hdu = fits.open('/home/innereye/astro/data/M83/jw02219-o011_t002_miri_f2100w_i2d.fits')
# # image = hdu[1].data.copy()
# # ##
# # lims = [0.03, 0.98]
# # hist_dat = image.flatten()
# # hist_dat = hist_dat[hist_dat > 0]
# # # nonzeros = np.abs(np.sign(fits_arr))
# # minval = np.quantile(hist_dat, lims[0])
# # maxval = np.quantile(hist_dat, lims[1])
# #     # minval, maxval = np.quantile(hist_dat, lims)
# # rescaled = (image - minval) / (maxval - minval)
# # rescaled_no_outliers = np.maximum(rescaled, np.quantile(rescaled, 0.002))
# # rescaled_no_outliers = np.minimum(rescaled_no_outliers, np.quantile(rescaled_no_outliers, 1.0 - 0.002))
# #
# #
# # ##
# # number_bins=10000
# # image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
# # cdf = image_histogram.cumsum()  # cumulative distribution function
# # cdf = cdf / cdf[-1]  # normalize
# #     # use linear interpolation of cdf to find new pixel values
# # image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
# # imageq = image_equalized.reshape(image.shape)
# # import matplotlib.pyplot as plt
# #
# # # Load and display the image
# # image = plt.imread('/media/innereye/My Passport/Data/JWST/SPT2147-50/rgb.png')
# # plt.imshow(image)
# # plt.axis('off')
# #
# # # Create a list to store the coordinates of the mouse clicks
# # click_coordinates = []
# #
# #
# # # Define a function to handle mouse click events
# # def onclick(event):
# #     if len(click_coordinates) < 2:
# #         # Append the coordinates of the clicked point to the list
# #         click_coordinates.append((event.xdata, event.ydata))
# #
# #         # Plot a red dot at the clicked point
# #         plt.plot(event.xdata, event.ydata, 'ro')
# #         plt.draw()
# #
# #         if len(click_coordinates) == 2:
# #             # After collecting two points, close the figure to proceed
# #             plt.close()
# #
# #
# # # Connect the onclick function to the figure
# # plt.connect('button_press_event', onclick)
# #
# # # Display the figure
# # plt.show()
# #
# # # Retrieve the coordinates of the two mouse clicks
# # p1, p2 = click_coordinates
# #
# # # Crop the image based on the selected rectangle
# # x1, y1 = int(min(p1[0], p2[0])), int(min(p1[1], p2[1]))
# # x2, y2 = int(max(p1[0], p2[0])), int(max(p1[1], p2[1]))
# # cropped_image = image[y1:y2, x1:x2]
# #
# # # Display the cropped image
# # plt.imshow(cropped_image)
# # plt.axis('off')
# # plt.show()
# #
# # # (token, frob) = flickr.get_token_part_one(perms='write')
# # # auth_url = flickr.auth_url(token)
# #
# #
# # # from astro_list_ngc import ngc_html_thumb
# # # ngc_html_thumb()
# # #
# # # from astro_list_ngc import make_thumb
# # # import os
# # # os.chdir('/media/innereye/My Passport/Data/JWST/data/NGC-7469-MRS/')
# # # make_thumb('NGC-7469-MRS_MIRI.png', '2022-07-04')
# # # from astro_list_ngc import remake_thumb
# # # remake_thumb()
