from astro_utils import *
os.chdir('/home/innereye/Pictures/ngc602')
# auto_plot('NGC-602', exp='*.fits', png='fac4nf.png', pkl=True, resize=False, method='rrgggbb', blc=False,
#           plot=False, fill=False, deband=True, adj_args={'factor': 4}, crop=False, whiten=False)
imgc = plt.imread('fac4nf.png')[..., :3]  # no fill image, filling holes created huge blue and green issues
#
mxrg = np.max([imgc[..., 0], imgc[..., 1]], 0)
mxrb = np.max([imgc[..., 0], imgc[..., 2]], 0)

## fix blue
# imgc = plt.imread('fac4nf.png')[..., :3]
blue = imgc[..., 2]
mask = ((imgc[..., 2] > 70/255) & (imgc[..., 2]/mxrg > 1.5)) | (mxrg < 0.05)
blue[mask] = mxrg[mask]
## fix green
green = imgc[..., 1]
mask = ((imgc[..., 1] > 70/255) & (imgc[..., 1]/mxrb > 1.5)) | (mxrb < 0.05)
green[mask] = mxrb[mask]
plt.imsave('masks.png', imgc)
##
# mxs = [mxrg, mxrb]
# ico = [2, 1]
mxmin = 0.05  # 0.05
colmin = [40, 40, 40]  # 70
ratio = [4, 1.5, 1.5]
imgc = plt.imread('fac4nf.png')[..., :3]
for ii in range(3):
    baseline = [0, 1, 2]
    baseline.pop(ii)
    baseline = np.array(baseline)
    mx = np.max(imgc[..., baseline], 2)
    col = imgc[..., ii]
    mask = ((imgc[..., ii] > colmin[ii] / 255) & (imgc[..., ii] / mx > ratio[ii])) | (mx < mxmin)
    col[mask] = mx[mask]
    imgc[..., ii] = col
plt.imsave('masks_3.png', imgc)
