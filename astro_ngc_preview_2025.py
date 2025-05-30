import os
from astro_utils import *
from astropy.time import Time
from astro_list_ngc import choose_fits, make_thumb, ngc_html_thumb
from glob import glob
from astro_ngc_align import add_crval_to_logs
from atproto import Client as Blient
from atproto import client_utils, models
from mastodon_bot import connect_bot

blient = Blient()
blient.login(os.environ['Bluehandle'], os.environ['Blueword'])
blim = 250  # should be 300 limit but failed once
masto, loc = connect_bot()

def post_image(message, image_path, mastodon=True, bluesky=True):
    # toot = f"\U0001F916 image processing for NASA / STScI #JWST \U0001F52D data ({target}). RGB Filters: {filt_str}"
    # toot = toot + f"\nPI: {info['PI_NAME']}, program {info['PROGRAM']}. CRVAL: {np.round(hdr0['CRVAL1'], 6)}, {np.round(hdr0['CRVAL2'], 6)}"
    post = {}
    if bluesky:
        boot = client_utils.TextBuilder()
        txt = message
        if len(txt) > blim:
            txt = txt[:blim]
        boot.text(txt)

        try:
            with open(image_path, 'rb') as f:
                img_data = f.read()
            alt = image_path.split('/')[-1]
            post['bsky'] = blient.send_image(text=boot, image=img_data, image_alt=alt)
            # success += 'bsky;'
            print('boot image')
        except:
            print(f'failed bluesky post for {image_path}')
    if mastodon:
        try:
            metadata = masto.media_post(image_path, "image/jpeg")
            post['masto'] = masto.status_post(message, media_ids=metadata["id"])
            print('toot image')
        except:
            print(f'failed mastodon post for {image_path}')
    return post


def reply_to_post(post, text=None, link=None):
    """Reply with a link to a post with an image."""
    if text is None:
        text = '123 test'
    boot = client_utils.TextBuilder()
    boot = boot.text(text)
    if link is not None:
        boot.link(link[0], link[1])
    parent = models.create_strong_ref(post)
    root = models.create_strong_ref(post)
    rep = blient.send_post(
        text=boot,
        reply_to=models.AppBskyFeedPost.ReplyRef(parent=parent, root=root)
    )
    return rep
##
df = pd.read_csv('ngc.csv', sep=',')
# drive = '/media/innereye/KINGSTON/JWST/'
##
if os.path.isdir('/home/innereye'):
    path2astro ='/home/innereye/astro'
    path2logs = path2astro+'/logs/'
    path2thumb = path2astro+'/docs/thumb/'
    if not os.path.isdir(drive):
        raise Exception('where is the drive?')
else:
    print(os.getcwdb())
    drive = os.getcwdb().decode('utf-8')  # os.environ['HOME']+'/astro'
    path2logs = drive+'/logs/'
    path2thumb = drive+'/docs/thumb/'
    path2astro = drive
os.chdir(drive)
##
last_post_row = np.where((df['posted'].str.contains('http')) | (df['posted'].values == 'failed'))[0][0]
if last_post_row == 0:
    print(f"last NGC already posted ({df['target_name'][last_post_row]})")
else:
    for row in range(last_post_row):
        pkl = False
        tgt = df['target_name'][row]
        try:
            os.chdir(drive)
            date0 = df["collected_from"][row][:10]
            log_csv = f'{path2logs}{tgt}_{date0}.csv'
            already = glob('/home/innereye/astro/docs/thumb/'+date0+'_'+tgt+'*')
            forbidden = False
            if 'TRAPEZIUM-CLUSTER-P1_2022-09-26.csv' in log_csv:  # too large
                forbidden = True
            if forbidden:
                print('forbidden: ' + date0 + '_' + tgt)
            else:
                both_apart = False  # no overlap between MIRI and NIRCam
                if tgt == 'NGC-6822-MIRI-TILE-2-COPY' or tgt == 'NGC-346-TILE-6':
                    both_apart = True
                if len(already) > 0:
                    msg = 'pictures exist:    '
                    for alr in already:
                        msg += '\n    ' + alr.split('/')[-1]
                    print(msg)
                elif 'background' in tgt.lower() or 'BKG' in tgt:
                    print('no background for now '+tgt)
                else:
                    if not os.path.isdir('data'):
                        os.system('mkdir data')
                    if not os.path.isdir('data/' + tgt):
                        os.system('mkdir data/' + tgt)
                    if os.path.isfile(log_csv):
                        download_by_log(log_csv, tgt=tgt)
                        chosen_df = pd.read_csv(log_csv)
                    else:
                        t_min = [np.floor(Time(df['collected_from'][row]).mjd),
                                 np.ceil(Time(df['collected_to'][row]).mjd)]
                        args = {'target_name': tgt,
                                't_min': t_min,
                                'obs_collection': "JWST",
                                'calib_level': 3,
                                'dataRights': 'public',
                                'intentType': 'science',
                                'dataproduct_type': "image"}
                        table = Observations.query_criteria(**args)
                        files = list(table['obs_id'])
                        files = [x + '_i2d.fits' for x in files]
                        print(f'[{row}] downloading {tgt} by query')
                        download_fits_files(files, destination_folder='data/' + tgt, wget=False)
                        chosen_df = choose_fits(files, folder='data/' + tgt)
                        if all(chosen_df['chosen']):
                            filtnum = filt_num(chosen_df['file'].values)
                            order = np.argsort(filtnum)
                            chosen_df = chosen_df.iloc[order]
                        else:
                            chosen_chosen = chosen_df[chosen_df['chosen'] == True]
                            not_chosen = chosen_df[chosen_df['chosen'] == False]
                            df2 = [chosen_chosen, not_chosen]
                            for idf in [0, 1]:
                                filtnum = filt_num(df2[idf]['file'].values)
                                order = np.argsort(filtnum)
                                df2[idf] = df2[idf].iloc[order]
                            chosen_df = pd.concat(df2)
                        chosen_df.to_csv(log_csv, index=False, sep=',')
                    ## make image
                    # read the files and for each filter, choose smaller and close to target images
                    use = chosen_df['chosen'].to_numpy()
                    files = np.asarray(chosen_df['file'])
                    # see if we have both MIRI and NIRCam, choose RGB method accordingly
                    files = files[use]
                    filt = filt_num(files)
                    files = files[np.argsort(filt)]
                    filt = np.sort(filt)
                    mn = np.zeros((len(files),2))
                    for ii in range(len(files)):
                        if 'miri' in files[ii]:
                            mn[ii,0] = 1
                        if 'nircam' in files[ii]:
                            mn[ii,1] = 1
                        if mn[ii, :].sum() == 0:
                            print('no miri and no nircam '+files[ii])
                            mn[ii, :] = np.nan
                        elif mn[ii, :].sum() == 2:
                            raise Exception('both miri and nircam')
    
                    method = 'rrgggbb'
                    if np.nanmean(mn[:, 0]) == 1:
                        instrument = 'MIRI'
                    elif np.nanmean(mn[:,1]) == 1:
                        instrument = 'NIRCam'
                    else:
                        instrument = 'NIRCam+MIRI'
                        method = 'mnn'
                    os.chdir('data')# os.chdir('..')
                    plotted = []
                    # TODO decide if to use 0.5 1 1
                    made_png = False
                    if np.nansum(mn[:, 0]) >= 2:
                        auto_plot(tgt, exp=list(files[mn[:, 0] == 1]), png=tgt+'_MIRI.jpg', pkl=pkl, method='rrgggbb', fill=True, adj_args={'factor':1})
                        plotted.append(tgt+'_MIRI.jpg')
                        made_png = True
                    if np.nansum(mn[:,1]) >= 2:
                        img = auto_plot(tgt, exp=list(files[mn[:, 1] == 1]), png=tgt + '_NIRCam.jpg', pkl=pkl, method='rrgggbb', fill=True, adj_args={'factor':1}, deband=True)
                        img = grey_zeros(img, replace=np.max)
                        plt.imsave(tgt + '_NIRCam.jpg', img, origin='lower')
                        plotted.append(tgt + '_NIRCam.jpg')
                        made_png = True
                    if '+' in instrument and np.nansum(mn) >= 2 and not both_apart:
                        if not os.path.isfile('nooverlap.txt'):
                            try:
                                auto_plot(tgt, exp=list(files[~np.isnan(mn[:, 0])]), png=tgt+'_'+instrument+'.jpg', pkl=pkl, method='mnn', fill=True, adj_args={'factor':1})
                                plotted.append(tgt+'_'+instrument+'.jpg')
                                made_png = True
                            except:
                                print('no overlap???????????????')
                                # os.system('echo "no overlap" > nooverlap.txt')
                    ##
                    if made_png:
                        path2images = make_thumb(plotted, date0, path2thumb=path2thumb)
                        ipic = -1
                        for pic in plotted:
                            ipic += 1
                            message = f"testing new code, processing JWST STScI data for {tgt} {pic.split('_')[-1][:-4]}"
                            post = post_image(message, path2images[ipic])
                            if 'masto' in post.keys():
                                url = post['masto']['url']+';'
                            else:
                                url = ''
                            if 'bsky' in post.keys():
                                url = url + 'https://bsky.app/profile/astrobotjwst.bsky.social/post/'+post['uri'].split('/')[-1] + ';'
                            url = url[:-1]
                        print('DONE ' + date0 + '_' + tgt)
                        df.at[row, 'posted'] = url
                        df.to_csv(path2astro+'/ngc.csv', index=False)
                    else:
                        print('no plots for '+ date0 + '_' + tgt)
                        df.at[row, 'posted'] = 'failed'
                        df.to_csv(path2astro+'/ngc.csv', index=False)
        except Exception as error:
            print('FAILED '+tgt)
            print(error)
            break
    add_crval_to_logs(path2astro=path2astro, drive=drive)


# print('trying sending to oshi')
# err = os.system(f"curl -T {pic} https://oshi.ec > tmp.txt")
# if err:
#    print('error sending to oshi')
#else:
# print('sent to oshi')
#with open('tmp.txt', 'r') as tmp:
#    dest = tmp.read()
#download_link = dest.split('\n')[2].split(' ')[0]
#print(dest)
#print(f"sent {pic} to: {download_link}")
# text = 'A high resolution image will be available for a few days at '
# link = ['https://oshi.ec', download_link]
# rep = reply_to_post(post, text, link)
