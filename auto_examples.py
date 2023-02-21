from astro_utils import auto_plot


auto_plot('ngc1672', '*_i2d.fits', png=True, pow=[0.5, 1, 1], core=True)
auto_plot('ngc1672', '*_i2d.fits', png='red_sqrt.png', method='mnn', pow=[0.5, 1, 1], pkl=True)

# auto_plot('ngc3256', '*miri*w_i2d.fits')
auto_plot('ngc1512', '*_i2d.fits',  png=True, pow=[0.75, 1, 1], resize=True, method='mnn')
auto_plot('ngc1512', '*_i2d.fits',  png=True, pow=[0.5, 1, 1], core=True)

auto_plot('ngc1672', '*_i2d.fits', png=True, pow=[0.5, 1, 1], method='mnn', resize=True)

auto_plot('NGC4321', '*miri*_i2d.fits', png=True, pow=[0.5, 1, 1], core=True)


auto_plot('ngc3627', '*_i2d.fits', png=True, pow=[0.5, 1, 1], method='mnn', pkl=True, resize=True)
# auto_plot('ngc3627', '*nircam*_i2d.fits', png=True, pow=[0.5, 1, 1], core=True)

auto_plot('ngc1300', '*_i2d.fits', png=True, pow=[0.5, 1, 1], method='mnn', pkl=True, resize=True)
auto_plot('ngc1300', '*_i2d.fits', png=True, pow=[0.5, 1, 1], method='mnn', pkl=True, core=True)

auto_plot('ngc1566', '*_i2d.fits', png=True, pow=[0.5, 1, 1], method='mnn', pkl=True, resize=True)

auto_plot('ngc1385', '*_i2d.fits', png=True, pow=[0.5, 1, 1], method='mnn', pkl=True, resize=True)

auto_plot('M16', '*_i2d.fits', png=True, pow=[0.5, 1, 1], pkl=True, resize=True)

auto_plot('ngc2070', '*_i2d.fits', png=True, pow=[0.5, 1, 1], pkl=True, resize=False)

auto_plot('vv114', '*_i2d.fits', png=True, pow=[1,1.5,1], factor=4, pkl=True)

auto_plot('ngc7496', '*_i2d.fits', png=True, pow=[0.5,1,1], factor=4, pkl=True, method='mnn', resize=True)