#!/usr/bin/env python

import random
import re


def pick(filenames, k=1, randomized=True):
    """Pick some thread files from a thread directory."""
    if k < 0:
        raise Exception('k should be non-negative')
    if randomized:
        random.shuffle(filenames)
    else:
        pattern = '([0-9]+)\.txt'
        filenames.sort(key=lambda f: int(re.search(pattern, f).group(1)))
    return filenames[:k]


def expected(support, pmf):
    """Compute expected value of a discrete RV with given support and pmf."""
    expval, cum = 0.0, 0.0
    for k, p in zip(support, pmf):
        if p < 0:
            raise Exception('Probability must be nonnegative')
        cum += p
        expval += k * p
    if abs(cum - 1.0) >= 1e-7:
        raise Exception('Probability distribution must sum to one')
    return expval


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
