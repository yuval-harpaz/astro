from mastodon import Mastodon

#   Set up Mastodon
mastodon = Mastodon(
    access_token='token.secret',
    api_base_url='https://botsin.space/'
)

mastodon.status_post("script test")
