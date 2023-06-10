from mastodon import Mastodon
import os
import sys


def connect_bot():
    local = '/home/innereye/astro/'
    if os.path.isdir(local):
        os.chdir(local)
        oauth = 'token.secret'
        loc = 'local'
    else:
        oauth = os.environ['OAuth']
        loc = 'github'
    masto = Mastodon(
        access_token=oauth,
        api_base_url='https://botsin.space/'
    )
    return masto, loc


if __name__ == "__main__":
    masto, loc = connect_bot()
    # metadata = masto.media_post("pics/NGC1365_miri.png", "image/png")
    # masto.status_post('toot image test '+loc, media_ids=metadata["id"])
    masto.status_post(sys.argv[1])
