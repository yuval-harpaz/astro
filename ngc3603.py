from astro_utils import *
from astro_fill_holes import *
img = auto_plot('NGC3603-EMPT+TA+SPITZER', exp='log', png='max_color.png', pow=[1, 1, 1], pkl=False, method='rrgggbb', resize=False, plot=True, adj_args={'ignore0': True, 'factor': 1}, max_color=True)
filled = img.copy()
for lay in range(3):
    xy = hole_xy(img[..., lay])
    size = hole_size(img[..., lay], xy, plot=False)
    filled[..., lay] = hole_disk_fill(img[..., lay], xy, size+3)

plt.imsave('filled3.png', filled, origin='lower')
# from astro_list_ngc import
# auto_plot('NGC-346', exp='*w_i2d.fits', png='all.png', pow=[1, 1, 1], pkl=True, method='mnn', crop=False, plot=False)
# auto_plot('ORIBAR-IMAGING-NIRCAM', exp='*_f*.fits', png='f.png', pow=[1, 1, 1], pkl=False, crop=True, method='rrgggbb')
# hdu = fits.open('/home/innereye/astro/data/M83/jw02219-o011_t002_miri_f2100w_i2d.fits')
# image = hdu[1].data.copy()
# ##
# lims = [0.03, 0.98]
# hist_dat = image.flatten()
# hist_dat = hist_dat[hist_dat > 0]
# # nonzeros = np.abs(np.sign(fits_arr))
# minval = np.quantile(hist_dat, lims[0])
# maxval = np.quantile(hist_dat, lims[1])
#     # minval, maxval = np.quantile(hist_dat, lims)
# rescaled = (image - minval) / (maxval - minval)
# rescaled_no_outliers = np.maximum(rescaled, np.quantile(rescaled, 0.002))
# rescaled_no_outliers = np.minimum(rescaled_no_outliers, np.quantile(rescaled_no_outliers, 1.0 - 0.002))
#
#
# ##
# number_bins=10000
# image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
# cdf = image_histogram.cumsum()  # cumulative distribution function
# cdf = cdf / cdf[-1]  # normalize
#     # use linear interpolation of cdf to find new pixel values
# image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
# imageq = image_equalized.reshape(image.shape)
# import matplotlib.pyplot as plt
#
# # Load and display the image
# image = plt.imread('/media/innereye/My Passport/Data/JWST/SPT2147-50/rgb.png')
# plt.imshow(image)
# plt.axis('off')
#
# # Create a list to store the coordinates of the mouse clicks
# click_coordinates = []
#
#
# # Define a function to handle mouse click events
# def onclick(event):
#     if len(click_coordinates) < 2:
#         # Append the coordinates of the clicked point to the list
#         click_coordinates.append((event.xdata, event.ydata))
#
#         # Plot a red dot at the clicked point
#         plt.plot(event.xdata, event.ydata, 'ro')
#         plt.draw()
#
#         if len(click_coordinates) == 2:
#             # After collecting two points, close the figure to proceed
#             plt.close()
#
#
# # Connect the onclick function to the figure
# plt.connect('button_press_event', onclick)
#
# # Display the figure
# plt.show()
#
# # Retrieve the coordinates of the two mouse clicks
# p1, p2 = click_coordinates
#
# # Crop the image based on the selected rectangle
# x1, y1 = int(min(p1[0], p2[0])), int(min(p1[1], p2[1]))
# x2, y2 = int(max(p1[0], p2[0])), int(max(p1[1], p2[1]))
# cropped_image = image[y1:y2, x1:x2]
#
# # Display the cropped image
# plt.imshow(cropped_image)
# plt.axis('off')
# plt.show()
#
# # (token, frob) = flickr.get_token_part_one(perms='write')
# # auth_url = flickr.auth_url(token)
#
#
# # from astro_list_ngc import ngc_html_thumb
# # ngc_html_thumb()
# #
# # from astro_list_ngc import make_thumb
# # import os
# # os.chdir('/media/innereye/My Passport/Data/JWST/data/NGC-7469-MRS/')
# # make_thumb('NGC-7469-MRS_MIRI.png', '2022-07-04')
# # from astro_list_ngc import remake_thumb
# # remake_thumb()
