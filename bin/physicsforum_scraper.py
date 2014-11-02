#!/usr/bin/env python

import argparse
import os
import os.path

from bs4 import BeautifulSoup
import requests


def post_filter(tag):
    if tag.name != 'blockquote':
        return False
    if not tag.has_attr('class'):
        return False
    return isinstance(tag['class'], list) and 'messageText' in tag['class']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape Physicsforum forum')
    parser.add_argument('id', type=str, help='Thread ID')
    parser.add_argument('-p', '--page', type=int, default=1,
                        help='Page number')
    parser.add_argument('-o', '--outdir', type=str, help='Output directory')
    parser.add_argument('-n', '--number', type=int, default=0,
                        help='Start post number')
    args = parser.parse_args()

    url = 'https://www.physicsforums.com/threads/{}/page-{}'
    url = url.format(args.id, args.page)
    r = requests.get(url)
    if r.status_code == 200:
        # Create save directory
        if args.outdir is not None:
            savedir = os.path.join(args.outdir, args.id)
        else:
            savedir = args.id
        os.makedirs(savedir, exist_ok=True)

        soup = BeautifulSoup(r.text)
        for i, post in enumerate(soup.find_all(post_filter)):
            num = i + args.number
            savefile = os.path.join(savedir, 'post-{}.txt'.format(num))
            with open(savefile, 'w') as fout:
                print('\n'.join(list(post.stripped_strings)), file=fout)
