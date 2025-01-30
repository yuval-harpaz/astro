from astro_utils import *
os.chdir('/media/innereye/KINGSTON/JWST/data/SN2023ixf')
files = sorted(glob('*1500*.fits') + glob('*2100*.fits'))
# o040 is from 2024-03-15, o003 is from 2025-01-27
plt.figure()
sp = 0
for file in files:
    data = fits.open(file)
    img = level_adjust(data[1].data)
    print(img.shape)
    sp += 1
    plt.subplot(2,2,sp)
    plt.imshow(img)
    plt.title(file)



##
for ii in [0, 2]:
    files2 = files[ii:ii+2]
    print(files2)
    data = fits.open(files2[1])
    red = data[1].data - 230
    red = red / 2500 / 2
    red[red < 0] = 0
    red[ red > 1] = 1
    data = fits.open(files2[0])
    blue = data[1].data - 39
    blue = blue / 2300
    blue[blue < 0] = 0
    blue[blue > 1] = 1
    green = (red + blue)/2
    img = np.zeros((red.shape[0], red.shape[1], 3))
    img[..., 0] = red
    img[..., 1] = green
    img[..., 2] = blue
    img[np.isnan(img)] = 0
    # img = log(img) * 27
    img = img * 100
    img[img > 1] = 1
    plt.figure()
    plt.imshow(img, origin='lower')

    if ii == 0:
        year = '2024-03-15'
    else:
        year = '2025-01-27'
    plt.imsave(year+'.png', img)


for ii in [0, 2]:
    files2 = files[ii:ii+2]
    print(files2)
    data = fits.open(files2[1])
    red = level_adjust(data[1].data, factor=1)
    data = fits.open(files2[0])
    blue = level_adjust(data[1].data, factor=1)
    green = (red + blue)/2
    img = np.zeros((red.shape[0], red.shape[1], 3))
    img[..., 0] = red
    img[..., 1] = green
    img[..., 2] = blue
    # img[np.isnan(img)] = 0
    # img = log(img) * 27
    # img = img * 100
    # img[img > 1] = 1
    plt.figure()
    plt.imshow(img, origin='lower')

    if ii == 0:
        year = '2024-03-15'
    else:
        year = '2025-01-27'
    plt.imsave(year+'.png', img)


# plt.imsave('Mgray1log.jpg', img, origin='lower',  pil_kwargs={'quality': 95}, cmap='gray')

# os.chdir('/media/innereye/My Passport/Data/JWST/data/NGC0300MIRI/')
# path = glob('*fits')
# layers = reproject(path)
# for ii in [0, 1]:
#     layers[..., ii] = level_adjust(layers[..., ii])


# plt.figure()
# plt.imshow(layers[..., [1, 0, 0]])

# rgb = layers[..., [1, 0, 0]].copy()
# for row in range(rgb.shape[0]):
#     vec = rgb[row, :, 0].copy()
#     vec = vec[vec > 0]
#     rgb[row, :, 0] = rgb[row, :, 0] - np.median(vec)
#     print(row)

# rgb[..., 0] = rgb[..., 0] + np.min(rgb[..., 0])
# rgb[..., 0] = level_adjust(rgb[..., 0])
# rgb[..., 1] = (rgb[..., 0] + rgb[..., 2]) / 2
# plt.figure()
# plt.imshow(rgb)

# plt.imsave('ngc300.png', rgb, origin='lower')
