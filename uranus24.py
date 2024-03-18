# import os

from astro_utils import *
import cv2
##
os.chdir('/media/innereye/My Passport/Data/JWST/Uranus24hr/')
layers = auto_plot('Uranus24hr', exp='logUranus24all.csv', png='filt_all1.png', pkl=False, resize=False,
                   method='filt', blc=True, opvar='layers',
                   plot=False, fill=True, deband=False, adj_args={'factor': 1}, whiten=False,
                   crop='y1=1862; y2=2382; x1=1880; x2=2397')
with open('layers12.pkl', 'wb') as f:
    pickle.dump(layers, f)



for ii in range(12):
    layers[:, :, ii] = level_adjust(layers[:, :, ii], factor=1)

lay4 = np.zeros((layers.shape[0], layers.shape[1], 4))
for ii in range(4):
    lay4[:, :, ii] = np.min(layers[:, :, ii*3:ii*3+3], axis=2)

# for ii in range(4):
#     lay4[:, :, ii] = level_adjust(lay4[:, :, ii], factor=1)
filt = [182, 210, 410, 480]
col = matplotlib.cm.jet(filt / np.max(filt))[:, :3]
rgb = assign_colors(lay4, col)
rgb = blc_image(rgb)
plt.figure()
plt.imshow(rgb, origin='lower')

##
layers = np.load('layers12.pkl', allow_pickle=True)
layers = layers[150:370, 120:365, :]
for ii in range(12):
    layers[:, :, ii] = level_adjust(layers[:, :, ii], factor=1)

lay4 = np.zeros((layers.shape[0], layers.shape[1], 4))
for ii in range(4):
    lay4[:, :, ii] = np.min(layers[:, :, ii*3:ii*3+3], axis=2)
rgb = assign_colors(lay4, col)
rgb = blc_image(rgb)

plt.figure()
plt.imshow(rgb, origin='lower')
plt.figure()
for ii in range(3):
    rgb = assign_colors(layers[..., ii::4], col)
    rgb = blc_image(rgb)
    if ii == 2:
        rgb[:,:,0] = rgb[:,:,0]
    plt.subplot(1,3,ii+1)
    plt.imshow(rgb, origin='lower')

for jj in range(12):
    plt.subplot(4,3,jj+1)
    plt.imshow(layers[..., jj], cmap='gray', origin='lower')
    plt.axis('off')

##
os.chdir('/media/innereye/My Passport/Data/JWST/Uranus24hr/')
layers = np.load('layers12.pkl', allow_pickle=True)
layers = layers[150:370, 120:365, :]
for ii in range(12):
    layers[:, :, ii] = level_adjust(layers[:, :, ii], factor=1)
font = cv2.FONT_HERSHEY_SIMPLEX
org = (1, 35)
fontScale = 1
color = (255, 255, 255)
thickness = 1
filt = [210, 410, 480]
col = matplotlib.cm.jet(filt / np.max(filt))[:, :3]

# plt.figure()
for ii in range(3):
    rgb = assign_colors(layers[..., ii+3::3], col)
    rgb = blc_image(rgb)
    rgb = rgb[::-1, ...]*255
    # rgb = whiten_image(rgb)
    rgb = cv2.putText(rgb.astype('uint8'), f'{ii+6}.2', org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    # plt.subplot(1,3,ii+1)
    # plt.imshow(rgb)
    if ii > 0:
        shift = 1
        rgb[:, shift:, :] = rgb[:, :-shift, :]
    plt.imsave(f'clouds00{ii}.png', rgb)
for rep in range(4):
    for ii in range(3):
        os.system(f'cp clouds00{ii}.png clouds0{str(ii+rep*3).zfill(2)}.png')
os.system('rm clouds.mp4')
os.system('ffmpeg -r 0.5 -f image2 -i clouds%3d.png clouds.mp4')

