from astro_utils import *
# from astropy.io import ascii
# table = ascii.read('tmp.csv')
# table = table.sort(keys='t_obs_release')
n = 14
print(f'reading {n} days')
table = last_n_days(n=n, html=False, products=False)

end_time = Time.now().mjd
start_time = end_time - n
table = Observations.query_criteria(obs_collection="JWST",
                                                t_obs_release=[start_time, end_time],
                                                calib_level=3,
                                                dataproduct_type="image")
if len(table) > 0:
    table = table[table['intentType'] == 'science']
    if len(table) > 0:
        table = table[table['dataRights'] == 'PUBLIC']
        if len(table) > 0:
            tdf = table.to_pandas()
            table = table[list(~tdf['obs_title'].str.contains("alibration"))]


if len(table) == 0:
    raise Exception('no new images')
table = table.to_pandas()
table = table.sort_values('t_obs_release', ascending=False, ignore_index=True)
page = '<!DOCTYPE html>\n<html>\n<head>\n  <title>Image Display Example</title>\n  <style>\n   img {\n      max-width: 19vw; /* Limit image width to P% of viewport width */\n      height: auto; /* Maintain aspect ratio */\n    }\n  </style>\n</head>\n<body>'
page += '<h1>JWST science images by release date (' + str(n)+ ' days)</h1><h2>by <a href="https://twitter.com/yuvharpaz" target="_blank">@yuvharpaz</a><br>'
date_prev = ''
print('making html')
for iimg in range(len(table)):  # min([len(table), n])):
    time = Time(table['t_obs_release'][iimg], format='mjd').utc.iso
    date = time[:10]
    if date != date_prev:
        page = page + '\n<br>' +date + '<br>\n'
    jpg = table['jpegURL'][iimg].replace('mast:JWST/product/', '')
    desc = 'title: ' + table['obs_title'][iimg] + '\n' + \
           'target: ' + table['target_name'][iimg] + '\n' + \
           'proposal: ' + str(table['proposal_id'][iimg]) + '\n' + \
           jpg + '\n' + time[:16]
    date_prev = date

    page = page + '\n<img src="https://mast.stsci.edu/portal/Download/file/JWST/product/' + jpg + f'" title="{desc}">'
page = page + '\n</body>\n</html>\n'
with open('docs/news_by_date.html', "w") as text_file:
    text_file.write(page)

# mir = plt.imread('/home/innereye/JWST/ngc_2835/ngc2835_miri.png')
# nir = plt.imread('/home/innereye/JWST/ngc_2835/ngc2835_nircam.png')
# mir = rgb2cmyk(mir)
# nir = rgb2cmyk(nir)
# mix = np.clip(nir+mir, 0, 255)  # .astype('uint8')
# mix = cmyk2rgb(mix).astype('uint8')
# plt.imshow(mix)
#
# ##
# pow = 0.75
# r = np.zeros((100,100,3))
# r[40:60,40:60,0] = pow
# b = np.zeros((100,100,3))
# b[50:70,50:70,2] = pow
# mix = 1 - (((1-r)**2 + (1-b)**2)/2)**0.5
# # mix = 1 - (((1-r)**2 + (1-b)**2)/3)
# plt.imshow(mix)
# #
# # c = rgb2cmyk(r)
# # fill = 0
# # c[np.isnan(c)] = fill
# # y = rgb2cmyk(b)
# # y[np.isnan(y)] = fill
# # # miks = c
# # # miks[..., :3] = c[..., :3] + y[..., :3]
# # # mix = cmyk2rgb(miks)
# # mix = 255-y-c
# # mix = np.clip(mix, 0, 255).astype('uint8')
# #