import pandas as pd
from matplotlib import pyplot as plt
import os
from glob import glob
import numpy as np
from skimage.transform import resize
import cv2
from astro_utils import blc_image, whiten_image


galaxy = [628, 1087, 1300, 1433, 1512, 1559, 1566, 1672, 2835, 3351, 3627, 4254, 4321, 4535, 5068, 5332, 7496, 1385]


os.chdir('/home/innereye/astro/docs/thumb/')
files = np.asarray(glob('*NIRCam+MIRI.png'))
for glx in galaxy:
    idx = [str(glx) in x for x in files]
    if np.sum(idx) == 0:
        print(f'no thmb for {glx}')
    elif np.sum(idx) > 1:
        print(f'more than one for {glx}')
# files = np.asarray([x for x in glob('ngc*.png') if '_' not in x])
df = pd.DataFrame(galaxy, columns=['ngc'])
df['h'] = 0
df['w'] = 0
df['fn'] = ''
for ii in range(len(galaxy)):
    glx = galaxy[ii]
    idx = [str(glx) in x for x in files]
    df.at[ii, 'fn'] = files[idx][0]
    img = plt.imread(df['fn'][ii])[:, :, :3]
    df.at[ii, 'h'] = img.shape[0]
    df.at[ii, 'w'] = img.shape[1]

prev = 0
for ii in range(18):
    wsum = prev + df['w'][ii]
    if wsum > 1920:
        print(f'{ii} images give {prev}')
        prev = 0
    else:
        prev = wsum

##
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
color = (255, 255, 255)
thickness = 1
sepw = 2
whiten = True
org = (17, 17)
##
collage = np.zeros((1200, 2024, 3), 'uint8')
prev = 0
row = 0
for ii in range(len(galaxy)):
    img = plt.imread(df['fn'][ii])[:, :, :3]
    if whiten:
        img = whiten_image(img)
    img = img*255
    img = img.astype('uint8')
    img = cv2.putText(img, df['fn'][ii].split('_')[1], org, font, fontScale, color, thickness, cv2.LINE_AA)
    wsum = prev + df['w'][ii]
    collage[300 * row:300 * row + 300, prev:wsum, :] = img
    if wsum > 1920:
        print(f'{ii} images give {prev}')
        print(f'row width {prev + df["w"][ii]}')
        prev = 0
        wsum = prev + df['w'][ii]
        row += 1
    else:
        # collage[300 * row:300 * row + 300, prev:wsum, :] = img
        prev = wsum
    # print(galaxy[ii])
    # print(img.shape)
collage = collage[:, :1942, :]
plt.imshow(collage)
plt.imsave('collage_18w.png', collage)
