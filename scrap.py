from astro_utils import *
auto_plot('ORIBAR-IMAGING-NIRCAM', exp='*_f*.fits', png='f.png', pow=[1, 1, 1], pkl=False, crop=True, method='rrgggbb')


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
