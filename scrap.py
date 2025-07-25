from astro_utils import *
ngc = pd.read_csv('ngc.csv')
missing_class = ngc['target_classification'].isnull()
not_missing_jpg = ~ngc['jpeg'].isnull()
for ii in np.where(missing_class & not_missing_jpg)[0]:
    fits = ngc['jpeg'][ii].split('/')[-1][:-4]+'.fits'
    program = str(ngc['proposal'][ii])
    obs_table = Observations.query_criteria(
        obs_collection="JWST",
        proposal_id=program,
        calib_level=3
    )
    for jj in range(len(obs_table)):
        found = False
        if fits in obs_table['dataURL'][jj]:
            ngc.at[ii, 'target_classification'] = obs_table['target_classification'][jj]
            found = True
            break
    if not found:
        print(f"No classification for {ii} ({ngc['target_name'][ii]})")
ngc.to_csv('ngc.csv', index=False) 

# auto_plot('SF_reg_1', exp='*.fits', method='filt05', png='filt2.jpg', crop=False, func=None, adj_args={'factor':2}, fill=True, deband=False, deband_flip=None, pkl=True)
# exp = ['jw06778-o001_t001_nircam_clear-f277w_i2d.fits', 'jw06778-o001_t001_nircam_clear-f335m_i2d.fits', 'jw06778-o001_t001_nircam_f444w-f470n_i2d.fits']
# auto_plot('SF_reg_1', exp=exp, method='rrgggbb', png='small2rgb.jpg', crop=False, func=None, adj_args={'factor':2}, fill=True, deband=False, deband_flip=None, pkl=False)
# exp = ['jw06778-o001_t001_nircam_clear-f090w_i2d.fits', 'jw06778-o001_t001_nircam_clear-f187n_i2d.fits', 'jw06778-o001_t001_nircam_clear-f200w_i2d.fits']
# auto_plot('SF_reg_1', exp='*clear*.fits', method='rrgggbb', png='reproj227_rgb.jpg', crop=False, func=None, adj_args={'factor':2}, fill=False, deband=False, deband_flip=None, pkl=True, reproject_to='f277w')

