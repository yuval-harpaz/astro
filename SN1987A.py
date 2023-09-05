from astro_utils import *
os.chdir('/media/innereye/My Passport/Data/JWST/data/SN-1987A')

nircam = glob('jw01726*.fits')
miri = glob('*brightsky*.fits')
auto_plot('SN-1987A', exp=miri+nircam[:5], png='clear.png', pkl=False, resize=False, method='rrgggbb', plot=False,
           max_color=False, fill=False, deband=False, adj_args={'factor': 4})

auto_plot('SN-1987A', exp=miri+nircam[5:], png='cropped.png', pkl=False, resize=False, method='rrgggbb', plot=False,
           max_color=False, fill=False, deband=False, adj_args={'factor': 4}, crop=True)

##
auto_plot('SN-1987A', exp=nircam[:5], png='clear_cropped2lims.png', pkl=False, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=False, deband=False, adj_args={'factor': 2, 'lims': [0.01, 0.99]}, crop='y1=467; y2=729; x1=485; x2=720')
##
auto_plot('SN-1987A', exp=nircam[5:], png='cropped2lims.png', pkl=False, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=False, deband=False, adj_args={'factor': 2, 'lims': [0.5, 0.99]}, crop='y1=467; y2=729; x1=485; x2=720')

img1 = plt.imread('clear_cropped2lims.png')
img2 = plt.imread('cropped2lims.png')
img = (img1+img2)/2
plt.imsave('merge.png', img)
