from astro_utils import *

auto_plot('NGC1559', exp='log', png='nircam_deband.png', pkl=True, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=False, deband=2, adj_args={'factor': 3})

# os.chdir('/media/innereye/My Passport/Data/JWST/data/HH211NIRCAM/')

rgb = auto_plot('NGC1559', exp='log', png='nircam_deband.png', pkl=True, resize=False, method='rrgggbb', plot=False,
          max_color=False, fill=False, deband=2, adj_args={'factor': 3})

remake = rgb.copy()
remake[..., 0] = np.max([rgb[..., 0], np.min(rgb[..., 1:], 2)], 0)


##
rgb = auto_plot('NGC1559', exp='log', png='deband_w.png', pkl=True, resize=False, method='mnnw', plot=False,
          max_color=False, fill=False, deband=False, adj_args={'factor': 3})

##