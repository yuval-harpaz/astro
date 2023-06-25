import pandas as pd
from glob import glob
from astro_utils import *
from astropy.time import Time
from mastodon_bot import connect_bot
from pyongc import ongc
##

def list_ngc():
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
    isori = [x[:3].upper() == 'ORI' for x in table['target_name']]
    ism = [x[0] == 'M' and x[1:].replace('-','').isnumeric() for x in table['target_name']]
    isic = [x[:2].upper() == 'IC' for x in table['target_name']]
    misc = ['Cartwheel', 'Comet', 'Antennae', 'Hoag', 'Arp', 'Pinwheel',
            'Sombrero', 'Sunflower', 'Tadpole', 'MESSIER', 'Whirlpool']
    ismisc = np.zeros(len(isngc), bool)
    for ix, x in enumerate(table['target_name']):
        for msc in misc:
            if msc.upper() in x.upper():
                ismisc[ix] = True
    # tm = table[ism].to_pandas()
    # m_ngc = []
    # for ii in range(len(tm)):
    #     m_ngc.append(ongc.get(tm['target_name'][ii].replace('-','')).name)


    table = table[np.array(isngc) | np.array(ism) | np.array(isori) | np.array(isic) | np.array(ismisc)]
    target_name = np.asarray(table['target_name'])
    target = np.unique(target_name)
    ##
    ngc = []
    for tt in target:
        if tt[0] == 'M':
            m = ongc.get(tt.replace('-', ''))
            if m is None:
                raise Exception('unable to find which ngc is: '+tt)
            else:
                ngc.append(int(m.name[3:]))
        elif tt[:2] == 'IC':
            ic = ongc.get(tt.replace('-', ''))
            if ic is None:
                raise Exception('unable to find which ngc is: ' + tt)
            if ic.name[:3] == 'NGC':
                ngc.append(int(ic.name[3:]))
            else:
                ngc.append(0)
        elif tt[:3] == 'NGC':
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
        elif tt[:3] == 'ORI':
            ngc.append(1976)
        else:
            ngc.append(0)
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
                session.append(np.arange(gap[iss-1]+1, gap[iss]+1))
            session.append(np.arange(gap[-1]+1, len(df1)))
        # if len(session) > 2:
        #     print(session)
        #     raise Exception('make sure sessions are fit')
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
    return df
##


def social(add=''):
    soc = '<div class="social-links"><h2>By yuval\n    ' \
             '<a href="https://nerdculture.de/@yuvharpaz"><img src="mastodona.png" alt="nerdculture.de/@yuvharpaz" /></a>\n    ' \
             '<a href="https://twitter.com/yuvharpaz"><img src="twitter-icon.png" alt="twitter.com/yuvharpaz" /></a>\n    ' \
             '<a href="https://github.com/yuval-harpaz"><img src="github-mark.png" alt="github.com/yuval-harpaz" /></a>\n    ' \
             + add + \
             '.    For updates on new images follow the bot    ' \
             '<a href="https://botsin.space/@astrobot_jwst"><img src="camelfav.ico" alt="botsin.space/@astrobot_jwst" /></a>    ' \
             '</h2>\n</div>'
    return soc


def credits():
    cre = '<div class="social-links"><h2>background inamge from\n    ' \
          '<a href="https://www.flickr.com/photos/nasawebbtelescope/52969542198/in/dateposted/">' \
          '<img src="flickr.png" alt="NASA (JWST) on Flicker">' \
          '</a>\n    </h2>\n</div>'
    return cre


def ngc_html():
    df = pd.read_csv('ngc.csv')
    for ii in [0, 1]:
        if ii == 0:
            vw = str(70)
            img_br = '<br>'
            grid = ''
            other = '<a href="https://yuval-harpaz.github.io/astro/ngc_grid.html" target="_blank">grid view</a>'
        else:
            vw = str(35)
            img_br = ''
            grid = '_grid'
            other = '<a href="https://yuval-harpaz.github.io/astro/ngc.html" target="_blank">stream view</a>'
        page = '<!DOCTYPE html>\n<html>\n<head>\n  ' \
               '<link rel="stylesheet" href="styles.css">' \
               '<title>JWST NGC images</title></title><link rel="icon" type="image/x-icon" href="camelfav.ico" />\n  ' \
               '<style>\n' \
               'img {\n      ' \
                   'max-width: ' + vw + 'vmin;\n      ' \
                   'height: auto;\n    }\n    ' \
               '</style>\n</head>\n<body><div class="container">'
        page = page + '<h1>JWST images of NGC objects, from latest to oldest release</h1>' \
                      'Preview images are the bluest (shortest wavelength)<br>'
        add = '<a href="https://github.com/yuval-harpaz/astro/blob/main/ngc.csv" target="_blank"> table</a>\n    '
        add = add + other
        page = page + social(add=add)
        for iimg in range(len(df)):  # min([len(tbl), n])):
            date = df.iloc[iimg]['release_date']
            # ngc = df.iloc[iimg]['NGC']
            tgt = df.iloc[iimg]['target_name']
            flt = df.iloc[iimg]['filters']
            desc = f'{date} {tgt}, available filters: [{flt}]'
            if ii == 0:
                page = page + f'\n<h3>{desc}</h3>'
            jpg = df['jpeg'].iloc[iimg].replace('mast:JWST/product/', '')
            page = page + '\n<img src="https://mast.stsci.edu/portal/Download/file/JWST/product/' + jpg + f'" title="{desc}">{img_br}'
        page = page + '\n</div></body>\n</html>\n'
        with open(f'docs/ngc{grid}.html', "w") as text_file:
            text_file.write(page)

def ngc_html_thumb():
    df = pd.read_csv('ngc.csv')
    # other = '<a href="https://yuval-harpaz.github.io/astro/ngc_thumb.html" target="_blank">stream view</a>'
    page = '<!DOCTYPE html>\n<html>\n<head>\n  ' \
           '<link rel="stylesheet" href="blackstyle.css">' \
           '<title>JWST NGC color images</title></title><link rel="icon" type="image/x-icon" href="camelfav.ico" />\n  ' \
           '<style>\n' \
           '</style>\n</head>\n<body><div class="container">'
    page = page + '<h1>A preview of JWST images of NGC objects, automatically colored using available filters</h1>' \
                  'Image triplets are NIRCam, NIRCam+MIRI, MIRI.  For NIRCam+MIRI images red = MIRI. The point is to make a fast, automatic process with fixed parameters. No manual touch, so alignment issues are expected.<br>'
    add = '<a href="https://github.com/yuval-harpaz/astro/blob/main/ngc.csv" target="_blank"> table</a>\n    '
    # add = add + other
    page = page + social(add=add)
    thumbs = np.sort(glob('/home/innereye/astro/docs/thumb/*.png'))[::-1]
    session_time = [x.split('/')[-1] for x in thumbs]
    target_name = np.asarray([x.split('_')[1] for x in session_time])
    instrument = np.asarray([x.split('_')[2][:-4] for x in session_time])
    session_time = np.asarray([x.split('_')[0] for x in session_time])
    for iimg in range(len(df)):  # min([len(tbl), n])):
        date = df.iloc[iimg]['collected_from'][:10]
        tgt = df.iloc[iimg]['target_name']
        idx = np.where((session_time == date) & (target_name == tgt))[0]
        if len(idx) > 0:
            # ngc = df.iloc[iimg]['NGC']
            flt = df.iloc[iimg]['filters']
            desc = f'{date} {tgt}, available filters: [{flt}]'
            page = page + f'\n<h3>{desc}</h3>'
            for jdx in idx:
                png = 'thumb/'+thumbs[jdx].split('/')[-1]
                tit = tgt+' '+instrument[jdx]
                page = page + '\n<img src="' + png + f'" title="{tit}">'
            page += '<br>\n'
    page = page + '\n</div></body>\n</html>\n'
    with open(f'docs/ngc_thumb.html', "w") as text_file:
        text_file.write(page)
def choose_fits(file_names=None, folder=''):
    if len(folder) > 0 and folder[-1] not in '\/':
        folder += '/'
    if file_names is None:
        file_names = glob('*.fits')
    offset = np.zeros(len(file_names))
    width = np.zeros(len(file_names))
    height = np.zeros(len(file_names))
    for ifile in range(len(file_names)):
        hdu = fits.open(folder+file_names[ifile])
        offset[ifile] = np.max(np.abs([hdu[0].header['XOFFSET'], hdu[0].header['YOFFSET']]))
        # size[ifile] = hdu[0].header['SUBSIZE1'] * hdu[0].header['SUBSIZE2']
        width[ifile] = hdu[1].data.shape[1]
        height[ifile] = hdu[1].data.shape[0]
        hdu.close()
    size = width*height
    order = np.argsort(size)  # sort from small to large, prefer smaller, no background
    file_names_sorted = np.asarray(file_names)[order]
    offset = offset[order]
    filt = filt_num(file_names_sorted)  # get filter number
    filtu = np.unique(filt)
    use = np.ones(len(file_names_sorted), bool)
    if len(filtu) < len(file_names_sorted):
        for ii in range(len(filtu)):
            idx = np.where(filt == filtu[ii])[0]
            selected = np.argmin(offset[idx])  # select the one closest to the target
            for jj in range(len(idx)):
                if jj != selected:
                    use[idx[jj]] = False
    chosen = np.zeros(len(file_names), bool)
    for ifn, fn in enumerate(file_names):
        idx = np.where(file_names_sorted == fn)[0][0]
        chosen[ifn] = use[idx]
    df = pd.DataFrame(list(zip(file_names, width, height, offset, chosen)),
                      columns=['file', 'width', 'height', 'offset', 'chosen'])
    return df

if __name__ == "__main__":
    df = list_ngc()
    df_prev = pd.read_csv('ngc.csv', sep=',')
    df.to_csv('ngc.csv', sep=',', index=False)
    ngc_html()
    if df.iloc[0]['release_date'] > df_prev.iloc[0]['release_date']:
        toot = 'A new NGC image at https://yuval-harpaz.github.io/astro/ngc.html'
        masto, _ = connect_bot()
        masto.status_post(toot)
