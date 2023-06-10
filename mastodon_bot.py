from mastodon import Mastodon
import os
#   Set up Mastodon
local = '/home/innereye/astro/'
if os.path.isdir(local):
    os.chdir(local)
    oauth = 'token.secret'
    loc = 'local'
else:
    oauth = os.environ['OAuth']
    loc = 'github'
mastodon = Mastodon(
    access_token=oauth,
    api_base_url='https://botsin.space/'
)

mastodon.status_post('script test '+loc)
