#!/usr/bin/env python

import random
import re


def pick(filenames, k=None, randomized=True):
    """Pick some thread files from a thread directory."""
    if randomized:
        random.shuffle(filenames)
    else:
        pattern = '([0-9]+)\.txt'
        filenames.sort(key=lambda f: int(re.search(pattern, f).group(1)))
    return filenames if k is None else filenames[:k]
