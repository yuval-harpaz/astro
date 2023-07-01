# astro
imaging astronomy data from JWST. Python code to download and do some post-processing for *.fits files, coming from James Webb Space Telescope (JWST) via [mast](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html).
## NGC color preview images
NGC objects cover many galaxies and nebula, so here it just means interesting stuff. I allowed some objects that are not in the NGC catalogue to slip in.<br>
Images here were automatically created from the data. They are saved with low resolution, and meant to allow us to see what data has been collected. For high resolution images I still need to overcome coregistration issues, and fill holes. With code. 
[![Alt a preview page for most NGC objects captured by JWST](ngc_thumb.png)](https://yuval-harpaz.github.io/astro/ngc_thumb.html)
## NGC images by release date
This is a news page, updated every 2 hours, showing example images from the latest release (I took the highest wavelength filter). You can get notified for new releases by following my 
mastodon bot[![Alt mastodon bot](docs/camelfav.ico)](href="https://botsin.space/@astrobot_jwst)<br> 
[![Alt a preview page for most NGC objects captured by JWST](ngc_stream.png)](https://yuval-harpaz.github.io/astro/ngc.html)
[![Alt a preview page for most NGC objects captured by JWST](ngc_grid.png)](https://yuval-harpaz.github.io/astro/ngc_grid.html)
### installation
I followed the instructions recommended [here](https://github.com/spacetelescope/jdat_notebooks) or [here](https://spacetelescope.github.io/jdat_notebooks/install.html#install) but froze my requirements.txt just in case. Basically you need anaconda3 and astropy. I work with pycharm and matplotlib so no notebooks here.
## Gallery
See more images at [GALLERY.md](https://github.com/yuval-harpaz/astro/blob/main/GALLERY.md)<br>
Some examples below:
### Cartwheel Galaxy
get_cartwheel.py<br>
cartwheel_long.py
![Alt nircam](https://github.com/yuval-harpaz/astro/blob/main/pics/cartwheel_nircam.png?raw=true)
### NGC 628
manifest = download_fits('ngc 628', include=['_miri_', '_nircam_', 'clear'])<br>
ngc629.py
#### NIRCam
![Alt ngc628_nircam](https://github.com/yuval-harpaz/astro/blob/main/pics/NGC_628_nircam.png?raw=true)
#### MIRI
![Alt ngc628_miri](https://github.com/yuval-harpaz/astro/blob/main/pics/NGC_628_miri.png?raw=true)


