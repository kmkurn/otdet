#!/usr/bin/env python

import random
import re


def pick(filenames, k=None, randomized=True):
    """Pick some thread files from a thread directory."""
    if k is not None and k < 0:
        raise ValueError('k should be non-negative')
    if randomized:
        random.shuffle(filenames)
    else:
        pattern = '([0-9]+)\.txt'
        filenames.sort(key=lambda f: int(re.search(pattern, f).group(1)))
    return filenames if k is None else filenames[:k]


class lazyproperty:
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls):
        if instance is None:
            return self
        else:
            value = self.func(instance)
            setattr(instance, self.func.__name__, value)
            return value
