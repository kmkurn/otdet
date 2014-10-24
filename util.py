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

    def __init__(self, N=1):
        self.N = N           # of posts taken (top N list)
        self._M = None       # of all posts
        self._n = None       # of OOT posts
        self._numexpr = 0    # of experiments
        self._freq = defaultdict(int)
        self._result_list = []

    def add_result(self, result):
        """Add an experiment result to be evaluated."""
        self._numexpr += 1
        if self._M is None:
            self._M = self._n = 0
            k = 0
            for i, (_, oot) in enumerate(result):
                self._M += 1
                if oot:
                    self._n += 1     # OOT post found
                    k += 1 if i < self.N else 0
            # Count the number of OOT posts in top k
            self._freq[k] += 1
        else:
            self._result_list.append(result)

    @lazyproperty
    def baseline(self):
        """Return the baseline performance vector.

        The baseline is obtaining OOT posts by chance. Thus, the baseline
        performance vector is the probability distribution of a hypergeometric
        random variable denoting the number of OOT posts in the top N list.
        Vector length is n+1, with k-th element represents the probability of
        getting k OOT posts in the top N list.
        """
        if self._numexpr == 0:
            raise Exception('You should do at least one experiment.')
        rv = hypergeom(self._M, self._n, self.N)
        k = np.arange(0, self._n+1)
        return rv.pmf(k)

    @lazyproperty
    def get_performance(self):
        """Return the evaluation result in a performance vector."""
        if self._numexpr == 0:
            raise Exception('You should do at least one experiment.')

        for result in self._result_list:
            k = sum(oot for _, oot in islice(result, self.N))
            self._freq[k] += 1

        res = np.zeros(self._n+1)
        for k in range(self._n+1):
            res[k] = self._freq[k] / self._numexpr

        return res
