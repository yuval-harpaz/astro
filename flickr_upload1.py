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
upload_response = flickr.upload(filename=photo_path, title=title, description=description)
photo_id = upload_response['photoid']

photo_path = '/media/innereye/My Passport/Data/JWST/data/NGC-3324/NGC-3324_MIRI.png'
title = 'NGC-3324, MIRI'
description = 'Cosmic Cliffs in Carina Nebula'
upload_response = flickr.upload(filename=photo_path, title=title, description=description)
photo_id = upload_response['photoid']

search_results = flickr.photos.search(user_id='me', text=title, per_page=2)
photos = search_results.find('photos').findall('photo')
photo_id = photos[0].attrib['id']  # '53005401542'
photo_info = flickr.photos.getInfo(photo_id=photo_id)
photo_url = photo_info.find('photo').find('urls').find('url').text

sizes_response = flickr.photos.getSizes(photo_id=photo_id)
sizes = sizes_response.find('sizes').findall('size')
for s in sizes:
    if s.values()[0] == 'Original':
        url = s.values()[3]
        break


# original_size = next((size for size in sizes if size['label'] == 'Original'), None)
sizes_tree = etree.fromstring(sizes_response.encode('utf-8'))

https://live.staticflickr.com/65535/53006267025_dd14c02643_o_d.png
photo_info = flickr.photos.getInfo(photo_id='53006267025')
photo_url = photo_info['photo']['urls']['url'][0]['_content']
# (token, frob) = flickr.get_token_part_one(perms='write')
# auth_url = flickr.auth_url(token)



##
# obj = flickr.photos.search(text='NGC', per_page=5, extras='url_o')
# url = obj.get('url_o')
# print(url)
# def flickr_search(keyward):
#     obj = flickr.photos.search(text=keyward,
#                            tags=keyward,
#                            extras='url_c',
#                            per_page=5)
#
#     for photo in obj:
#         url=photo.get('url_c')
#         photos = ET.dump(obj)
#         print (photos)


# from astro_list_ngc import ngc_html_thumb
# ngc_html_thumb()
#
# from astro_list_ngc import make_thumb
# import os
# os.chdir('/media/innereye/My Passport/Data/JWST/data/NGC-7469-MRS/')
# make_thumb('NGC-7469-MRS_MIRI.png', '2022-07-04')
# from astro_list_ngc import remake_thumb
# remake_thumb()
