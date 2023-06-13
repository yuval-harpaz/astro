import os

a = os.system('wget -O tmp.jpg https://mast.stsci.edu/portal/Download/file/JWST/product/jw01523-o010_t010_miri_f1130w-sub64_i2d.jpg')
if a == 0:
    print('okay')
else:
    raise Exception(a)

##
os.chdir('/home/innereye/JWST/HALFRING/')
files = glob('jw*.fits')
# size = 500
# I = np.zeros((500, 500, 3))
for ii in [0, 1]:
    noise = fits.open(files[ii])
    signal = fits.open(files[ii+2])
    img = signal[1].data-noise[1].data
    img[img < 0] = 0
    signal[1].data = img
    signal.writeto(f'dif{ii}.fits', overwrite=True)
    # img = level_adjust(img[-size:,-size:])
    # I[..., ii*2] = img
# img[..., 1] = (img[..., 0]+img[..., 2])/2



