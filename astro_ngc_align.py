# import pandas as pd
# import os.path
'''
run astro_ngc_preview first to create logs
'''
# import os

# import pandas as pd

from astro_utils import *
def add_crval_to_logs():
    df = pd.read_csv('ngc.csv', sep=',')
    # df = list_ngc()
    # df.to_csv('ngc.csv', sep=',', index=False)
    # ngc_html()
    # masto, loc = connect_bot()
    ##
    # loc == 'local'
    for row in range(len(df)):  # np.where(df['NGC'] == 3627)[0]:
        # pkl = True
        tgt = df['target_name'][row]
        drive = '/media/innereye/My Passport/Data/JWST/'
        if os.path.isdir(drive):
            os.chdir(drive)
        else:
            raise Exception('where is the drive?')
            # os.system('mkdir data')
        if not os.path.isdir('data/'+tgt):
            raise Exception('no dir')
        date0 = df["collected_from"][row][:10]
        # already = glob('/home/innereye/astro/docs/thumb/'+date0+'_'+tgt+'*')
        # if len(already) > 0:
        #     msg = 'pictures exist:    '
        #     for alr in already:
        #         msg += '\n    ' + alr.split('/')[-1]
        #     print(msg)
        if tgt in ['NGC2506G31', 'NGC-5139']:  # clusters
            print('no clusters for now '+date0+' '+tgt)
        elif tgt == 'ORIBAR-IMAGING-MIRI':
            print(tgt + ' too messy, two sessions with strange overlap + NIRCam')
        elif 'background' in tgt.lower() or 'BKG' in tgt:
            print('no background for now '+tgt)
        else:
            log_csv = f'/home/innereye/astro/logs/{tgt}_{date0}.csv'
            if os.path.isfile(log_csv):
                chosen_df = pd.read_csv(log_csv)
                files = list(chosen_df['file'])
                # print(f'[{row}] downloading {tgt} by log')
                # download_fits_files(files, destination_folder='data/' + tgt)
                if 'CRVAL1' in chosen_df.columns:
                    print(f'{tgt} got CRVAL1')
                else:
                    chosen_df['CRVAL1'] = 0
                    chosen_df['CRVAL2'] = 0
                    for ii in range(len(files)):
                        if os.path.isfile(drive+'data/'+tgt+'/'+files[ii]):
                            hdu = fits.open(drive+'data/'+tgt+'/'+files[ii])
                            chosen_df['CRVAL1'].at[ii] = hdu[1].header['CRVAL1']
                            chosen_df['CRVAL2'].at[ii] = hdu[1].header['CRVAL2']
                        else:
                            if chosen_df.iloc[ii]['chosen']:
                                raise Exception('chosen file missing: '+tgt+'/'+files[ii])
                    if chosen_df['CRVAL1'].to_numpy().max() > 0:
                        chosen_df.to_csv(log_csv, index=False)

def fix_by_log(target='NGC-3132', data_dir='/media/innereye/My Passport/Data/JWST/test/'):
    logs = glob('logs/'+target+'*')
    df = pd.read_csv(logs[0])
    if len(logs) > 1:
        for log in logs[1:]:
            df = pd.concat([df, pd.read_csv(log)])
            df = df.reset_index(drop=True)
    if 'CRVAL1fix' in df.columns:
        files = glob(data_dir+target+'/*.fits')
        if len(files) > 0:
            for file in files:
                row = np.where(df['file'] == file.split('/')[-1])[0]
                if len(row) == 0:
                    print(file.split('/')[-1]+' not in logs')
                else:
                    fixed = False
                    hdu = fits.open(file)
                    for xy in [1, 2]:
                        crval = df[f'CRVAL{xy}fix'].values[row]
                        if ~np.isnan(crval):
                            hdu[1].header[f'CRVAL{xy}'] = crval[0]
                            fixed = True
                    if fixed:
                        hdu.writeto(file, overwrite=True)
                        print('fixed ', file)



if __name__ == '__main__':
    add_crval_to_logs()
