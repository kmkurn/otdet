#!/usr/bin/env python

import glob
import os.path
import random
import re


def pick(directory, k=None, randomized=True):
    """Pick some thread files from a thread directory."""
    all_files = glob.glob(os.path.join(directory, '*'))
    if randomized:
        random.shuffle(all_files)
    else:
        pattern = '([0-9]+)\.txt'
        all_files.sort(key=lambda f: int(re.search(pattern, f).group(1)))
    return all_files if k is None else all_files[:k]
