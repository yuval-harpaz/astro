import matplotlib.pyplot as plt

from astro_utils import *
os.chdir('/media/innereye/My Passport/Data/JWST/data/')
from cv2 import putText, FONT_HERSHEY_SIMPLEX, LINE_AA
for ii in range(5):
    fn = glob(f'PLUTO{ii}/*1500*.fits')[0]
    hdu = fits.open(fn)
    time = hdu[0].header['DATE-OBS']+' '+hdu[0].header['TIME-OBS'][:8]
    img = plt.imread(f'/home/innereye/Pictures/pluto/pluto{ii}.png')
    img = putText(img, time, (40, 40),
                  FONT_HERSHEY_SIMPLEX, 0.6,
                  1, 1, LINE_AA)
    plt.imsave(f'/home/innereye/Pictures/pluto/pluto_time{ii}.png', img)