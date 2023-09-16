''' last 3 layers '''
from astro_utils import *
##
auto_plot('LDN-694', exp='*.fits', png='nircam3.png', pkl=True, resize=True, method='rrgggbb', plot=False,
          max_color=False, fill=False, deband=True, adj_args={'factor': 3})
##
os.chdir('/media/innereye/My Passport/Data/JWST/data/LDN-694/')
##
# if os.path.isfile('baseline3.png'):
#     bl = plt.imread('baseline3.png')[..., :3]
# else:
#     layers = np.load('adjusted.pkl', allow_pickle=True)
#     layers = layers[..., 3:]
#     layers = layers[..., ::-1]
#     plt.figure()
#     plt.imshow(layers)
#     plt.imsave('baseline3.png', layers, origin='loewer')
bl = plt.imread('nircam3.png')[..., :3]
hsv = matplotlib.colors.rgb_to_hsv(bl)
# hue = hsv[..., 0]
##
plt.figure()
for row in np.arange(250, 600):
    idx = np.where((hsv[25:250, row, 2] > 0.2) & (hsv[25:250, row, 1] > 0.5))[0]
    plt.scatter(hsv[idx, row, 0].astype(float), hsv[idx, row, 2].astype(float), s=3, c=bl[idx, row, :])
    print(row)
    # for ii in idx:
    #     plt.scatter(hsv[ii, row, 0].astype(float), hsv[ii, row, 2].astype(float), s=3, c=bl[ii, row, :])
##
half_win = 0.05
# noise = (hue < hsv_win)  | (np.abs(hue - 1/3) < hsv_win) | (np.abs(hue - 2/3) < hsv_win)
bins = [-half_win, 1/3, 2/3]
print('make holes')
rgb = bl.copy()
for lay in range(bl.shape[2]):
    layer = bl[..., lay]
    noise = np.abs(hsv[..., 0] - bins[lay]) < half_win*2
    if lay == 0:
        noise = noise | (hsv[..., 0] > (1-half_win))
    noise = noise & (hsv[..., 1] > 0.5)
    noise = noise & (hsv[..., 2] > 0.25)
    layer[noise] = 0
    # layers[:, :, lay] = hole_conv_fill(layer, n_pixels_around=0, clean_below=0.01)
    # filled = hole_conv_fill(filled, n_pixels_around=3, ringsize=15, clean_below_local=0.75, clean_below=2)
    print(lay)
print('saving')
plt.imsave('holes3.png', bl)  # , origin='loewer')
##
print('conv fill')
# mx = np.nanmax(bl, 2)
holes = plt.imread('holes3.png')
for lay in range(holes.shape[2]):
    layer = holes[:, :, lay]
    holes[:, :, lay] = hole_conv_fill(layer, n_pixels_around=0, clean_below=0)
    print(lay)

plt.imsave('conv3.png', holes)
##
degreen = holes[..., :3].copy()
idx = np.argmax(degreen, axis=2) == 1
green = degreen[..., 1]
mx = np.max(degreen[..., [0, 2]], axis=2)
green[idx] = mx[idx]
##
deblue = degreen.copy()
idx = np.argmax(deblue, axis=2) == 2
blue = deblue[..., 2]
mx = np.max(deblue[..., [0, 1]], axis=2)
blue[idx] = mx[idx]
plt.imsave('red2white.png', deblue)
##
# rgb = np.zeros((bl.shape[0], bl.shape[1], 3))
# for ii in range(3):
#     rgb[..., ii] = np.nanmax(bl[..., ii*2:ii*2+2], 2)
# rgb = rgb[..., ::-1]
# plt.imsave('conv.png', rgb, origin='loewer')
#
example = np.zeros((1000, 3000, 3))
img = plt.imread('baseline3.png')
example[:,:1000,:] = img[2500:3500, 3500:4500, :3]
img = plt.imread('holes3.png')
example[:,1000:2000,:] = img[2500:3500, 3500:4500, :3]
img = plt.imread('conv3.png')
example[:,2000:,:] = img[2500:3500, 3500:4500, :3]
plt.imsave('example3.png', example)

#



#
