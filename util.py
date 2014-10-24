#!/usr/bin/env python

from collections import defaultdict
from itertools import islice
import random
import re

import numpy as np
from scipy.stats import hypergeom


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


class Evaluator:
    """Evaluate performance of OOT detector."""

    def __init__(self, N=1, M=None, n=None):
        self.N = N      # of posts taken (top N list)
        self.M = M      # of all posts
        self.n = n      # of OOT posts
        self.freq = defaultdict(int)
        self.result_list = []

    def add_result(self, result):
        """Add an experiment result to be evaluated."""
        if self.M is None:
            self.M = self.n = 0
            k = 0
            for i, (oot, distance) in enumerate(result):
                self.M += 1
                if oot:
                    self.n += 1     # OOT post found
                    k += 1 if i < self.N else 0
            # Count the number of OOT posts in top k
            self.freq[k] += 1
        else:
            self.result_list.append(result)

    @lazyproperty
    def baseline(self):
        """Return the baseline performance vector.

        The baseline is obtaining OOT posts by chance. Thus, the baseline
        performance vector is the probability distribution of a hypergeometric
        random variable denoting the number of OOT posts in the top N list.
        Vector length is n+1, with k-th element represents the probability of
        getting k OOT posts in the top N list.
        """
        if self.M is None:
            raise ValueError('Cannot determine total number of posts.')
        rv = hypergeom(self.M, self.n, self.N)
        k = np.arange(0, self.n+1)
        return rv.pmf(k)

    def evaluate(self):
        """Return the evalution result in a performance vector."""
        numexpr = 0     # of experiments
        for result in self.result_list:
            numexpr += 1
            k = sum(oot for oot, _ in islice(result, self.N))
            self.freq[k] += 1

        res = np.zeros(self.n+1)
        for k in range(self.n+1):
            res[k] = self.freq[k] / numexpr

        return res
