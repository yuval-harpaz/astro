from astroquery.mast import Observations
import os
if not os.path.isdir('data'):
    os.mkdir('data')
os.chdir('data')
if not os.path.isdir('cartwheel'):
    os.mkdir('cartwheel')
os.chdir('cartwheel')
obs_table = Observations.query_object("Cartwheel")
# get table rows where obs_collection="JWST"
obs_table = obs_table[obs_table["obs_collection"] == "JWST"]
# df = obs_table.to_pandas()  # watch as pandas table
for tt in obs_table:
    all = Observations.get_product_list(tt)
    filt = all[all["productType"] == "SCIENCE"]
    filt = filt[filt["dataRights"] == "PUBLIC"]
    filt = Observations.filter_products(filt, extension='_i2d.fits')
    if len(filt) > 0:
        islong = False
        for jj in filt:
            if 'long' in jj['obs_id']:  ## looking for a dataset where filenames contain "long"
                islong = True
        if islong:
            Observations.download_products(filt)
