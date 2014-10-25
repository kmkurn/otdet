#!/usr/bin/env python

import argparse
import os
import os.path

from bs4 import BeautifulSoup
import requests


def post_filter(tag):
    """Filter tag containing a post."""

    def only_one_postmsg_class(attr):
        if isinstance(attr, str):
            return attr == 'postmsg'
        return len(attr) == 1 and attr[0] == 'postmsg'

    return tag.name == 'div' and tag.has_attr('class') and\
        only_one_postmsg_class(tag['class'])


def text_filter(tag):
    """Filter tag containing text."""
    return tag.name == 'p' and not tag.has_attr('class')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scrape Arch Linux forum')
    parser.add_argument('id', type=str, help='Thread ID')
    parser.add_argument('-p', '--page', type=int, default=1,
                        help='Page number')
    parser.add_argument('-o', '--outdir', type=str, help='Output directory')
    parser.add_argument('-n', '--number', type=int, default=0,
                        help='Start post number')
    args = parser.parse_args()

    url = 'https://bbs.archlinux.org/viewtopic.php'
    payload = {'id': args.id, 'p': args.page}
    r = requests.get(url, params=payload)
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
                pstr = [p.stripped_strings for p in post.descendants
                        if text_filter(p)]
                for text in pstr:
                    print('\n'.join(text), file=fout)
