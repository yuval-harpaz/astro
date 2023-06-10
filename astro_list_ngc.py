from astro_utils import *
from astropy.time import Time
from mastodon_bot import connect_bot
##
args = {'obs_collection': "JWST",
        'calib_level': 3,
        'dataRights': 'public',
        'intentType': 'science',
        'dataproduct_type': "image"}

# search images by observation date and by release date
table = Observations.query_criteria(**args)
isnotnirspec = ['NIRSPEC' not in x.upper() for x in table['instrument_name']]
table = table[isnotnirspec]
isngc = [x[:3].upper() == 'NGC' for x in table['target_name']]
table = table[isngc]
target_name = np.asarray(table['target_name'])
target = np.unique(target_name)
ngc = []
for tt in target:
    nn = tt[3:]
    for ii, ll in enumerate(nn):
        if ll.isnumeric():
            break
    if ii > 1:
        raise Exception('expected numbers')
    nn = nn[ii:]
    for ii, ll in enumerate(nn):
        if not ll.isnumeric():
            break
    if ii == len(nn)-1:
        ii += 1
    ngc.append(int(nn[:ii]))
##
row = []
for ii, tt in enumerate(target):
    idx = np.where(target_name == tt)[0]
    df1 = table[idx].to_pandas()
    df1 = df1.sort_values('t_min', ignore_index=True)
    gap = np.where(np.diff(df1['t_min']) > 4)[0]

    if len(gap) == 0:
        session = [np.arange(len(df1))]
    else:
        session = [np.arange(gap[0]+1)]
        for iss in range(1, len(gap)):
            session.append(np.arange(gap[iss-1], gap[iss]+1))
        session.append(np.arange(gap[-1]+1, len(df1)))
    if len(session) > 2:
        print(session)
        raise Exception('make sure sessions are fit')
    for ses in session:
        df2 = df1.iloc[ses]
        df2 = df2.reset_index(drop=True)
        filt = filt_num(df2['dataURL'])
        bluer = np.argmin(filt)
        filt = str(np.unique(filt).astype(int))
        url = df2['jpegURL'][bluer]
        t_min = Time(df2['t_min'].iloc[0], format='mjd').utc.iso
        # t_min = t_min[:10]
        t_max = Time(df2['t_max'].iloc[-1], format='mjd').utc.iso
        # t_max = t_max[:10]
        release = Time(df2['t_obs_release'].iloc[0], format='mjd').utc.iso
        release = release[:10]
        row.append([release, t_min, t_max, ngc[ii], tt, filt[1:-1], url])
df = pd.DataFrame(row, columns=['release_date', 'collected_from', 'collected_to', 'NGC','target_name','filters','jpeg'])
df = df.sort_values('release_date', ignore_index=True, ascending=False)
##
df_prev = pd.read_csv('ngc.csv', sep=',')
df.to_csv('ngc.csv', sep=',', index=False)
if df.iloc[0]['release_date'] > df.iloc[0]['release_date']:
    toot = 'A new NGC image at https://yuval-harpaz.github.io/astro/ngc.html'
    masto = connect_bot()
    masto.status_post(toot)
    # a = os.system('wget -O tmp.jpg ' + page[first_image + 10:page.index('.jpg') + 4])
    # if a == 0:
    #     img = plt.imread('tmp.jpg')
    #     size = os.path.getsize('tmp.jpg')
    #     mb2 = 2 * 10 ** 6  # mastodon allows 2MB
    #     if size >= mb2:
    #         ratio = mb2 / size
    #         img = resize(img, (int(ratio ** 0.5 * img.shape[0]), int(ratio ** 0.5 * img.shape[1])))
    #         plt.imsave('tmp.jpg', img, cmap='gray')
    #     metadata = masto.media_post("tmp.jpg", "image/jpeg")
    #     masto.status_post(toot, media_ids=metadata["id"])
    # else:
    #     masto.status_post(toot)
##
df = pd.read_csv('ngc.csv')
page = '<!DOCTYPE html>\n<html>\n<head>\n  <title>JWST NGC images</title>\n  ' \
       '<style>\n   img {\n      ' \
       'max-width: 35vw; /* Limit image width to P% of viewport width */\n      ' \
       'height: auto; /* Maintain aspect ratio */\n    }\n    ' \
       '.container {\n      ' \
       'margin-left: 5%;\n    }\n' \
       '</style>\n</head>\n<body><div class="container">'
page = page + '<h1>JWST images of NGC objects, from latest to oldest release</h1>' \
              'Preview images are the bluest (shortest wavelength)<br>by <a href="https://twitter.com/yuvharpaz" target="_blank">@yuvharpaz</a>,' \
              ' <a href="https://github.com/yuval-harpaz/astro/blob/main/astro_list_ngc.py" target="_blank"> code,</a>' \
              ' <a href="https://github.com/yuval-harpaz/astro/blob/main/ngc.csv" target="_blank"> table.</a><br><br>'
for iimg in range(len(df)):  # min([len(tbl), n])):
    date = df.iloc[iimg]['release_date']
    # ngc = df.iloc[iimg]['NGC']
    tgt = df.iloc[iimg]['target_name']
    flt = df.iloc[iimg]['filters']
    page = page + f'\n<h3>{date} {tgt}, filters: [{flt}]</h3>'
    jpg = df['jpeg'].iloc[iimg].replace('mast:JWST/product/', '')
    page = page + '\n<img src="https://mast.stsci.edu/portal/Download/file/JWST/product/' + jpg + f'""><br>'
page = page + '\n</div></body>\n</html>\n'
with open('docs/ngc.html', "w") as text_file:
    text_file.write(page)
