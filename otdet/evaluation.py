"""
Out-of-topic post detection evaluation methods.
"""

from collections import Counter

import numpy as np
from scipy.stats import hypergeom, skew

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

        size_tuples = [(len(result), sum(is_oot for _, is_oot in result))
                       for result in results]
        if len(set(size_tuples)) > 1 or size_tuples[0][0] != self.M or \
                size_tuples[0][1] != self.n:
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

    @lazyproperty
    def baseline_skew(self):
        """Return the skewness measure of baseline."""
        rv = hypergeom(self.M, self.n, self.N)
        return float(rv.stats(moments='s'))

    def get_performance(self, results):
        """Return the evaluation result in a performance vector."""
        self._validate(results)

        num_expr = len(results)
        n = sum(is_oot for _, is_oot in results[0])
        top_oot_nums = [sum(is_oot for _, is_oot in result[:self.N])
                        for result in results]

        res = np.zeros(n+1)
        count = Counter(top_oot_nums)
        for k in range(n+1):
            res[k] = count[k] / num_expr
        return res

    def get_performance_skew(self, results):
        """Compute performance skewness unbiased estimate."""
        self._validate(results)
        top_oot_nums = [sum(is_oot for _, is_oot in result[:self.N])
                        for result in results]
        return skew(top_oot_nums, bias=False)
