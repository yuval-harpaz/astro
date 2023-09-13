# import os.path
# import pickle
#
# import matplotlib.pyplot as plt
# from matplotlib import colors
from astro_utils import *
os.chdir('/media/innereye/My Passport/Data/JWST/data/HH211NIRCAM/')


hdul = fits.open('jw01257-o003_t005_nircam_clear-f210m_i2d.fits')
##
os.chdir('/media/innereye/My Passport/Data/JWST/HH211/')
obs_table = Observations.query_object("HH211")
obj = obs_table[obs_table['instrument_name'] == 'NIRCAM/IMAGE']
obj = obj[obj['dataRights'] == 'PUBLIC']
obj = obj[obj['target_name'] == 'HH211NIRCAM']
isurl = np.zeros(len(obj), bool)
for iobs in range(len(isurl)):
    isurl[iobs] = type(obj['jpegURL'][iobs]) == np.str_
obj = obj[isurl]
##
address = 'https://mast.stsci.edu/portal/Download/file/'
for iobs in range(len(obj)):
    try:
        os.system('wget -O tmp.jpg '+address+obj['jpegURL'][iobs][5:])
        img = plt.imread('tmp.jpg')
        plt.subplot(4, 5, iobs+1)
        plt.imshow(img)
        plt.title(obj['filters'][iobs])
    except:
        print(iobs)



##
for obs in obj:
    all = Observations.get_product_list(obs)
    filt = Observations.filter_products(all, extension='_cal.fits')
    filt = filt[filt['productType'] == 'SCIENCE']
    Observations.download_products(filt)


##
path = list_files('/media/innereye/My Passport/Data/JWST/HH211/', search='*image*')
short = []
long = []
for p in path:
    if 'long' in p:
        long.append(p)
    else:
        short.append(p)

# if not os.path.isfile('IC348-MOSAIC.pkl'):
#     auto_plot('IC348-MOSAIC', exp='log', png='deband.png', pkl=True, resize=False, method='rrgggbb', plot=False,
#                max_color=False, fill=False, deband=True, adj_args={'factor': 2})
# ##
# if os.path.isfile('adjusted.pkl'):
#     layers = np.load('adjusted.pkl', allow_pickle=True)
# else:
#     layers = np.load('IC348-MOSAIC.pkl', allow_pickle=True)
#     for lay in range(layers.shape[2]):
#         layers[..., lay] = level_adjust(layers[..., lay], factor=2)
#     with open('adjusted.pkl', 'wb') as f:
#         pickle.dump(layers, f)
# ##
# notnan = ~np.any(np.isnan(layers), axis=2)
# not0 = np.all(layers > 0, axis=2)
# if os.path.isfile('med.pkl'):
#     med = np.load('med.pkl', allow_pickle=True)
# else:
#     med = np.nanmedian(layers, 2)
#     with open('med.pkl', 'wb') as f:
#         pickle.dump(med, f)
# ##
# hsv_win = 0.05
# bl = plt.imread('baseline.png')[..., :3]
# hue = matplotlib.colors.rgb_to_hsv(bl)[..., 0]
# noise = (hue < hsv_win)  | (np.abs(hue - 1/3) < hsv_win) | (np.abs(hue - 2/3) < hsv_win)
# # noise = noise & (med < 0.9)
# # noise = (np.abs(hue - 1/3) < hsv_win)
# bl0 = bl.copy()
# for b in range(3):
#     bb = bl0[..., b]
#     bb[noise] = 0
#
# print('fill holes')
# for lay in range(layers.shape[2]):
#     layer = layers[..., lay]
#     # noise0 = noise & (layer > 0.75)
#     layer[noise] = 0
#     layers[:, :, lay] = hole_conv_fill(layer, n_pixels_around=0, clean_below=0.01)
#     # filled = hole_conv_fill(filled, n_pixels_around=3, ringsize=15, clean_below_local=0.75, clean_below=2)
#     print(lay)
# print('saving')
# rgb = np.zeros((layers.shape[0], layers.shape[1], 3))
# for ii in range(3):
#     rgb[..., ii] = np.nanmax(layers[..., ii*2:ii*2+2], 2)
# rgb = rgb[..., ::-1]
# plt.imsave('conv.png', rgb, origin='loewer')
#
# example = np.zeros((1000, 3000, 3))
# example[:,:1000,:] = bl[2500:3500, 3500:4500, :]
# example[:,1000:2000,:] = bl0[2500:3500, 3500:4500, :]
# example[:,2000:,:] = rgb[2500:3500, 3500:4500, :]
# plt.imsave('example.png', example, origin='loewer')
# # plt.figure()
# # plt.imshow(example)



