import os

a = os.system('wget -O tmp.jpg https://mast.stsci.edu/portal/Download/file/JWST/product/jw01523-o010_t010_miri_f1130w-sub64_i2d.jpg')
if a == 0:
    print('okay')
else:
    raise Exception(a)
