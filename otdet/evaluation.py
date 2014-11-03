"""
Out-of-topic post detection evaluation methods.
"""

from collections import Counter

import numpy as np
from scipy.stats import hypergeom

from otdet.util import lazyproperty


class TopListEvaluator:
    """Evaluate performance of OOT detector based on ranked result list."""

    def __init__(self, M=1, n=1, N=1):
        self.M = M          # of all posts
        self.n = n          # of OOT posts
        self.N = N          # of posts taken (top N list)

    def _validate(self, results):
        """Validate a list of result.

        A list of result must have nonzero length and the same number of
        posts as specified when creating evaluator object.
        """
        if len(results) == 0:
            raise Exception('Results cannot be empty.')

        sizelist = [(len(result), sum(oot for _, oot in result))
                    for result in results]
        if len(set(sizelist)) > 1 or sizelist[0][0] != self.M or \
                sizelist[0][1] != self.n:
            raise Exception('Number of posts mismatch.')

    @lazyproperty
    def baseline(self):
        """Return the baseline performance vector.

        The baseline is obtaining OOT posts by chance. Thus, the baseline
        performance vector is the probability distribution of a hypergeometric
        random variable denoting the number of OOT posts in the top N list.
        Vector length is n+1, with k-th element represents the probability of
        getting k OOT posts in the top N list.
        """
        rv = hypergeom(self.M, self.n, self.N)
        k = np.arange(0, self.n+1)
        return rv.pmf(k)

    def get_performance(self, results):
        """Return the evaluation result in a performance vector."""
        self._validate(results)

        numexpr = len(results)
        n = sum(oot for _, oot in results[0])
        ntop = [sum(oot for _, oot in result[:self.N]) for result in results]

        res = np.zeros(n+1)
        count = Counter(ntop)
        for k in range(n+1):
            res[k] = count[k] / numexpr

        return res
