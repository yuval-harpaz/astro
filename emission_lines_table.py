#!/usr/bin/python
import pandas as pd
import json

url0 = 'https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra='
url1 = '&limits_type=0&low_w=2&upp_w=20000&unit=2&submit=Retrieve+Data&de=0&format=0&line_out=0&en_unit=0&output=0&bibrefs=1&page_size=15&show_obs_wl=1&show_calc_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&conf_out=on&term_out=on&enrg_out=on&J_out=on'

elements = pd.read_csv('https://gist.githubusercontent.com/GoodmanSciences/c2dd862cd38f21b0ad36b8f96b4bf1ee/raw/1d92663004489a5b6926e944c1b3d9ec5c40900e/Periodic%2520Table%2520of%2520Elements.csv')
elements = list(elements['Symbol'])
data = {}
for element in elements:
    print('getting '+element)
    df = pd.read_html(url0+element+url1)
    emissions = []
    if len(df) > 3:
        lines = df[3]['Observed Wavelength Vac (µm)'].to_numpy()
        # lines = lines[:np.where(lines == 'Observed Wavelength Air (nm)')[0][0]]
        for line in lines:
            if type(line) == str:
                if ' ' in line:
                    print(element+' ', line)
                else:
                    emissions.append(float(line))
    else:
        print('nothing for '+element)
    data[element] = emissions

with open('emission_lines_µm.json', 'w') as fp:
    json.dump(data, fp)
