# import os.path
# import pickle
#
# import matplotlib.pyplot as plt
# from matplotlib import colors
from astro_utils import *
os.chdir('/media/yuval/PNY/JWST/Mab')

files = sorted(glob('*444*.fits'))
end_time = []
num1 = []
for ii in range(len(files)):
    hdu = fits.open(files[ii])
    end_time.append(hdu[0].header['DATE-END'])
    num1.append(files[ii].split('_')[0].split('-')[1])
    print(f"{num1[-1]} {end_time[-1]}")

order = np.argsort(end_time)
end_time = np.array(end_time)[order]
num1 = np.array(num1)[order]
files = np.array(files)[order]
plt.figure()
for ii in range(len(end_time)):
    plt.subplot(2, 3, ii+1)
    hdu = fits.open(files[ii])
    plt.imshow(level_adjust(hdu[1].data), origin='lower')
    plt.title(num1[ii])
##
files = glob('*.fits')
for iobs in range(len(num1)):
    output = end_time[iobs][:10] + '_full.jpg'
    if not os.path.exists(output):
        files_obs = sorted([f for f in files if num1[iobs] in f])
        for ifile in [1, 0]:
            hdu = fits.open(files_obs[ifile])
            data = hdu[1].data - 0.25
            data = data/(1.4 - 0.25)
            data[data < 0] = 0
            data[data > 1] = 1
            if ifile == 1:
                header = hdu[1].header.copy()
                img = np.zeros((data.shape[0], data.shape[1], 3))
                img[..., 0] = data
            else:
                hdu[1].data = data
                reproj, _ = reproject_interp(hdu[1].copy(), header)
                img[..., 2] = reproj
            img[..., 1] = (img[..., 0] + img[..., 2])/2
        plt.imsave(output, img, origin='lower', pil_kwargs={'quality':95})

## sqrt
destination = [f for f in files if '444' in f and 'o004' in f][0]
hdu = fits.open(destination)
header = hdu[1].header.copy()
img = np.zeros(hdu[1].data.shape + (3,))
for iobs in range(len(num1)):
    output = end_time[iobs][:10] + '_sqrt.jpg'
    if iobs in [3, 4]:
        baseline = 0.4
    else:
        baseline = 0.25
    if iobs == 5:
        fac = 1.6
    else:
        fac = 1.4
    if not os.path.exists(output):
        files_obs = sorted([f for f in files if num1[iobs] in f])
        for ifile in [0, 1]:
            hdu = fits.open(files_obs[ifile])
            data = hdu[1].data - baseline
            data = data/(fac - baseline)
            data[np.isnan(data)] = 0
            data[data < 0] = 0
            data[data > 1] = 1
            data = data ** 0.5
            if files_obs[ifile] == destination:
                reproj = data
            else:
                hdu[1].data = data
                reproj, _ = reproject_interp(hdu[1].copy(), header)
            if ifile == 0:
                img[..., 2] = reproj
            else:
                img[..., 0] = reproj
            img[..., 1] = (img[..., 0] + img[..., 2])/2
        plt.imsave(output, img, origin='lower', pil_kwargs={'quality':95})
## percentiles
destination = [f for f in files if '444' in f and 'o004' in f][0]
hdu = fits.open(destination)
header = hdu[1].header.copy()
img = np.zeros(hdu[1].data.shape + (3,))
for iobs in range(len(num1)):
    output = end_time[iobs][:10] + '_prct.jpg'
    if not os.path.exists(output):
        files_obs = sorted([f for f in files if num1[iobs] in f])
        for ifile in [0, 1]:
            hdu = fits.open(files_obs[ifile])
            prct = np.nanpercentile(hdu[1].data, [1, 99.5])
            print(files_obs[ifile])
            # print(prct)
            baseline = prct[0]
            if baseline < 0.3:
                baseline = 0.3
            fac = prct[1]-baseline
            print(fac)
            data = hdu[1].data - baseline
            data = data/(fac - baseline)
            data[np.isnan(data)] = 0
            data[data < 0] = 0
            data[data > 1] = 1
            data = data ** 0.5
            if files_obs[ifile] == destination:
                reproj = data
            else:
                hdu[1].data = data
                reproj, _ = reproject_interp(hdu[1].copy(), header)
            if ifile == 0:
                img[..., 2] = reproj
            else:
                img[..., 0] = reproj
            img[..., 1] = (img[..., 0] + img[..., 2])/2
        plt.imsave(output, img, origin='lower', pil_kwargs={'quality':95})

# ffmpeg for collecting *_prct.jpg into mp4
os.system('ffmpeg -y -framerate 1 -pattern_type glob -i "*_prct.jpg" -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -crf 18 -pix_fmt yuv420p mab_prct.mp4')
#
jpgs = glob('*.jpg')
for ii in range(len(jpgs)):
    data = plt.imread(jpgs[ii]).copy().astype(float)
    if ii == 0:
        avg = data
    else:
        avg += data
avg = avg/len(jpgs)
avg = avg.astype('uint8')
plt.imsave('avg.jpg', avg, pil_kwargs={'quality':95})


jpgs = glob('*.jpg')
for ii in range(len(jpgs)):
    data = plt.imread(jpgs[ii]).copy().astype(float)
    data[data == 0] = np.nan
    if ii == 0:
        avg = data
        count = (~np.isnan(data)).astype(int)
    else:
        avg = np.nansum([avg, data], axis=0)
        count += (~np.isnan(data)).astype(int)
count[count == 0] = 1  # avoid division by zero
avg = avg/count
avg[np.isnan(avg)] = 0
avg = avg.astype('uint8')
plt.imsave('nanavg.jpg', avg, pil_kwargs={'quality':95})

## red-layer composite: avg of first 4 as red, 5th as green, 6th as blue
prct_jpgs = sorted(glob('*_prct.jpg'))  # 6 files sorted by date
red_layers = []
for f in prct_jpgs[:4]:
    r = plt.imread(f).copy().astype(float)[..., 0]
    r[r == 0] = np.nan
    red_layers.append(r)
red_avg = np.nanmean(red_layers, axis=0)
red_avg[np.isnan(red_avg)] = 0
green = plt.imread(prct_jpgs[4]).copy().astype(float)[..., 0]
blue  = plt.imread(prct_jpgs[5]).copy().astype(float)[..., 0]
composite = np.stack([red_avg, green, blue], axis=-1)
composite = np.clip(composite, 0, 255).astype('uint8')
plt.imsave('composite_rgb.jpg', composite, pil_kwargs={'quality':95})

#
# num = filt_num(files)
#
# layers = np.load('HH-903.pkl', allow_pickle=True)
# img = np.zeros((layers.shape[0], layers.shape[1], 3))
# img[..., 0] = level_adjust(layers[..., 3], factor=2)-level_adjust(layers[..., 2], factor=2)
# img[..., 1] = level_adjust(layers[..., 1], factor=2)-level_adjust(layers[..., 0], factor=2)
# img[..., 2] = level_adjust(layers[..., 0], factor=2)
#
# img[img < 0] = 0
# plt.imshow(img, origin='lower')
#
# plt.imsave('HH-903.jpg', img, origin='lower', pil_kwargs={'quality':95})
