"""One-time run of the HST bot (hst_bot.py) on the past week, as a test.

    python hst_bot_week.py                  # dry run: images and texts in data/tmp/hst_bot, nothing posted or logged
    python hst_bot_week.py --post           # post and log to docs/bot_hst_posts.csv (the bot then skips these)
    python hst_bot_week.py --post --only NGC-5905 LEDA-855412
"""
import argparse

import hst_bot

ap = argparse.ArgumentParser()
ap.add_argument('--days', type=float, default=8)
ap.add_argument('--post', action='store_true')
ap.add_argument('--only', nargs='*', help='target names')
a = ap.parse_args()
res = hst_bot.run(a.days, do_post=a.post, only=a.only, write_log=a.post)
print('\nsummary:')
print(res[['target_name', 'filters', 'labels', 'posted']].to_string())
