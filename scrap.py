import flickrapi
with open('flickr.secret') as f:
    lines = f.readlines()
api_key = lines[0].split(' ')[-1][:-1]
api_secret = lines[1].split(' ')[-1][:-1]
flickr = flickrapi.FlickrAPI(api_key, api_secret)
flickr.get_request_token(oauth_callback='oob')
auth_url = flickr.auth_url(perms='write')
flickr.get_access_token('967-609-138')

photo_path = '/media/innereye/My Passport/Data/JWST/data/NGC-3324/NGC-3324_NIRCam.png'
title = 'NGC-3324, NIRCam'
description = 'Cosmic Cliffs in Carina Nebula'
flickr.upload(filename=photo_path, title=title, description=description)
# (token, frob) = flickr.get_token_part_one(perms='write')
# auth_url = flickr.auth_url(token)


# from astro_list_ngc import ngc_html_thumb
# ngc_html_thumb()
#
# from astro_list_ngc import make_thumb
# import os
# os.chdir('/media/innereye/My Passport/Data/JWST/data/NGC-7469-MRS/')
# make_thumb('NGC-7469-MRS_MIRI.png', '2022-07-04')
# from astro_list_ngc import remake_thumb
# remake_thumb()
