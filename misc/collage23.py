from matplotlib import pyplot as plt
import os
from glob import glob
import numpy as np
from skimage.transform import resize
import cv2
from astro_utils import blc_image, whiten_image

os.chdir('/home/innereye/JWST/collage')
files = np.asarray(glob('ngc*.png'))
# files = np.asarray([x for x in glob('ngc*.png') if '_' not in x])
ok = np.asarray(['core' in x for x in files])
cores = files[ok]
galaxy = files[~ok]
galaxy = np.sort(galaxy)
h0 = [0, 0, 0, 360, 360, 360, 720, 720, 720]
w0 = [0, 640, 1280, 0, 640, 1280, 0, 640, 1280]
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
color = (1, 1, 1)
thickness = 1
sepw = 2


def make_collage(whiten=False):
    collage = np.zeros((1080, 1920, 3))
    for ii in range(9):
        img = plt.imread(galaxy[ii])[:, :, :3]
        img = resize(img, [1080/3, 1920/3])
        # if blc:
        #     img = blc_image(img)
        if whiten:
            img = whiten_image(img)
        corename = galaxy[ii][:-4]+'_core.png'
        if os.path.isfile(corename):
            org = (90, 17)
            core = plt.imread(corename)[:,:,:3]
            core = resize(core, np.round([172/2, 172/2]))
            if whiten:
                core = whiten_image(core)
            img[:core.shape[0], :core.shape[1],:] = core
            img[core.shape[0]+1:core.shape[0]+sepw,:core.shape[1]+sepw, :] = 1
            img[:core.shape[0]+sepw, core.shape[1]+1:core.shape[1]+sepw, :] = 1
        else:
            org = (17, 17)
        img = cv2.putText(img, galaxy[ii][:-4].replace('ngc', 'NGC '), org, font, fontScale, color, thickness, cv2.LINE_AA)
        collage[h0[ii]:h0[ii]+360, w0[ii]:w0[ii]+640, :] = img
        # print(galaxy[ii])
        # print(img.shape)
    plt.imshow(collage)
    plt.imsave(f'collage_w{int(whiten)}.png', collage)

if __name__ == '__main__':
    make_collage(whiten=False)
    make_collage(whiten=True)


