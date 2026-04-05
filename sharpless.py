from astro_utils import *
os.chdir('/media/yuval/KINGSTON/JWST/data/S305_offset')
files = glob('*.fits')
filtnum = filt_num(files)
files = np.array(files)[np.argsort(-filtnum)]
filtnum = filtnum[np.argsort(-filtnum)]
choice = files[filtnum < 300]
x = 13232
y = 14635
rgb = np.zeros((x, y, len(choice)))
for ii in range(len(choice)):
     hdu = fits.open(choice[ii])
     # print(hdu[1].data.shape)
     data = hole_func_fill(hdu[1].data[:x, :y], func='max')
     data = deband_layer(data)  #func=np.percentile
     rgb[..., ii] = log1(level_adjust(data, factor=1))
     print(ii)
rgb3 = assign_colors_by_filt(rgb, np.array(filtnum[filtnum<300]))
rgb3 = rgb3 * 255
rgb3 = rgb3.astype('uint8')
plt.imsave('small_1_log.jpg', rgb3, origin='lower', pil_kwargs={'compression': 'jpeg', 'quality': 95})
# plt.imsave('Rafgl.jpg', rgbr, pil_kwargs={'quality': 95},origin='lower')
choice = files[filtnum >= 300]
x = 6489
y = 7216
rgb = np.zeros((x, y, len(choice)))
for ii in range(len(choice)):
     hdu = fits.open(choice[ii])
     # print(hdu[1].data.shape)
     data = log1(hole_func_fill(hdu[1].data[:x, :y], func='max'))
     rgb[..., ii] = level_adjust(data, factor=1)
     print(ii)
rgb3 = assign_colors_by_filt(rgb, np.array(filtnum[filtnum>=300]))
rgb3 = rgb3 * 255
rgb3 = rgb3.astype('uint8')
plt.imsave('large_1_log.jpg', rgb3, origin='lower', pil_kwargs={'compression': 'jpeg', 'quality': 95})
