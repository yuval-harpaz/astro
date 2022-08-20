# astro
imaging astronomy data from JWST. Python code to download and do some post-processing for *.fits files, coming from James Webb Space Telescope (JWST) via [mast](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html).
### installation
I followed the instructions recommended [here](https://github.com/spacetelescope/jdat_notebooks) or [here](https://spacetelescope.github.io/jdat_notebooks/install.html#install) but froze my requirements.txt just in case. Basically you need anaconda3 and astropy. I work with pycharm and matplotlib so no notebooks here.
### Cartwheel Galaxy
get_cartwheel.py<br>
cartwheel_long.py
![Alt nircam](https://github.com/yuval-harpaz/astro/blob/main/pics/median_rgb.png?raw=true)
### NGC 628
manifest = download_fits('ngc 628', include=['_miri_', '_nircam_', 'clear'])
ngc629.py
![Alt ngc628_nircam](https://github.com/yuval-harpaz/astro/blob/main/pics/NGC_628_nircam.png?raw=true)
![Alt ngc628_miri](https://github.com/yuval-harpaz/astro/blob/main/pics/NGC_628_miri.png?raw=true)

