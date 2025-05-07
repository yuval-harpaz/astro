from astro_utils import *
os.chdir('/home/innereye/astro/data/SN2024VJM')
files = np.array(glob('*fits'))[[0,2,1]]
filt = filt_num(files)  # pickle is saved blue to red
layers = np.load('SN2024VJM.pkl', allow_pickle=True)

rgb = np.zeros_like(layers)
for ii in range(3):
    rgb[..., 2-ii] = level_adjust(layers[..., ii])
# plt.imshow(rgb)
colormap = 'jet'
plt.figure()
for ii in range(4):
    sb = [0, 0.25, 0.5, 1][ii]
    ass = assign_colors_by_filt(rgb.copy()[::-1, ...], filt, legend=True, subtract_blue=sb, colormap=colormap)
    plt.subplot(1,4,ii+1)
    plt.imshow(ass)
    plt.axis('off')
    plt.title(f"subtract blue={sb}")
    # blc = blc_image(ass)
    # plt.subplot(2,4,ii+1+4)
    # plt.imshow(blc, origin='lower')
    # plt.axis('off')
    # plt.title(f"subtract blue={sb}, blced")

# rgb = blc_image(rgb)

# plt.imshow(rgb, origin='lower')

