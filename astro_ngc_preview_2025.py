import os
from astro_utils import *
from astropy.time import Time
from astro_list_ngc import choose_fits, make_thumb, ngc_html_thumb
from glob import glob
from astro_ngc_align import add_crval_to_logs
##
df = pd.read_csv('ngc.csv', sep=',')
# drive = '/media/innereye/KINGSTON/JWST/'
##
if os.path.isdir('/home/innereye'):
    if not os.path.isdir(drive):
        raise Exception('where is the drive?')
    else:
        drive = os.environ['HOME']+'/astro'
os.chdir(drive)
# raise Exception('where is the drive?')
if not os.path.isfile('docs/latest.csv'):
    os.chdir(drive)
    # raise Exception('am I in astro?')
if not os.path.exists('data/tmp'):
    os.makedirs('data/tmp')
##

##
for row in range(len(df)):
    pkl = True
    tgt = df['target_name'][row]
    try:
        os.chdir(drive)
        date0 = df["collected_from"][row][:10]
        log_csv = f'/home/innereye/astro/logs/{tgt}_{date0}.csv'
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
                if not os.path.isdir('data/' + tgt):
                    os.system('mkdir data/' + tgt)
                if os.path.isfile(log_csv):
                    download_by_log(log_csv, tgt=tgt)
                    chosen_df = pd.read_csv(log_csv)
                    # if date0 == '2022-09-18':
                    #     prev = '/home/innereye/astro/logs/ORIBAR-IMAGING-MIRI_2022-09-11.csv'
                    #     prev = pd.read_csv(prev)
                    #     prev['chosen'] = False
                    #     prev['chosen'].at[np.where([prev['file'].str.contains('miri_f1500w')])[0][0]] = True
                    #     chosen_df = pd.concat([prev, chosen_df], ignore_index=True)
                    #     both_apart = True
                    # files = list(chosen_df['file'][chosen_df['chosen']])
                    # print(f'[{row}] downloading {tgt} by log')
                    # download_fits_files(files, destination_folder='data/' + tgt)
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
                    # files = list(table['dataURL'])
                    # files = [x.split('/')[-1] for x in files]
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

                    # os.chdir('data/'+tgt)
                # check if pkl was saved locally


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
                    auto_plot(tgt, exp=list(files[mn[:, 0] == 1]), png=tgt+'_MIRI.png', pow=[1, 1, 1], pkl=False, resize=True, method='rrgggbb', plot=False)
                    plotted.append(tgt+'_MIRI.png')
                    made_png = True
                if np.nansum(mn[:,1]) >= 2:
                    auto_plot(tgt, exp=list(files[mn[:, 1] == 1]), png=tgt + '_NIRCam.png', pow=[1, 1, 1], pkl=False, resize=True, method='rrgggbb', plot=False)
                    plotted.append(tgt + '_NIRCam.png')
                    made_png = True
                if '+' in instrument and np.nansum(mn) >= 2 and not both_apart:
                    if not os.path.isfile('nooverlap.txt'):
                        try:
                            auto_plot(tgt, exp=list(files[~np.isnan(mn[:, 0])]), png=tgt+'_'+instrument+'.png', pow=[1, 1, 1], pkl=True, resize=True, method='mnn', plot=False)
                            plotted.append(tgt+'_'+instrument+'.png')
                            made_png = True
                        except:
                            print('no overlap???????????????')
                            os.system('echo "no overlap" > nooverlap.txt')
                ##
                if made_png:
                    make_thumb(plotted, date0)
                    print('DONE ' + date0 + '_' + tgt)
                    os.system(f"curl -T {plotted} https://oshi.ec > tmp.txt")
                    with open('tmp.txt', 'r') as tmp:
                        dest = tmp.read()
                    print(f"sent file to: {dest}")
                else:
                    print('no plots for '+ date0 + '_' + tgt)
    except Exception as error:
        print('FAILED '+tgt)
        print(error)
        break

        # for ii in range(len(plotted)):
        #     img = plt.imread(plotted[ii])[..., :3]
        #     # edge = np.where(np.mean(np.mean(img, 2), 1))[0][0]
        #     new_height = 300
        #     ratio = new_height / img.shape[0]
        #     imgrs = resize(img, (new_height, int(ratio * img.shape[1])))
        #     plt.imsave('/home/innereye/astro/docs/thumb/'+date0+'_'+plotted[ii], imgrs, cmap='gray')

ngc_html_thumb()
add_crval_to_logs()
