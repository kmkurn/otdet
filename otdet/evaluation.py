"""
Out-of-topic post detection evaluation methods.
"""

from collections import Counter

import numpy as np
from scipy.stats import hypergeom

from otdet.util import lazyproperty


class TopListEvaluator:
    """Evaluate performance of OOT detector based on ranked result list."""

    def __init__(self, result, M=None, n=None, N=1):
        if N < 0:
            raise Exception('Cannot pick negative number of posts in top list')
        self.result = result
        self.N = N                          # of posts taken (top N list)
        if M is None or n is None:
            Mtemp, ntemp = self._get_nums()
            self.M = Mtemp if M is None else M
            self.n = ntemp if n is None else n
        else:
            # Check validity of M and n
            if M < n:
                raise Exception('M should never be less than n')
            self.M, self.n = M, n

    def _get_nums(self):
        """Get the number of all and OOT posts."""
        def get_num_oot(subresult):
            return sum(is_oot for _, is_oot in subresult)

        temp = [(len(subresult), get_num_oot(subresult))
                for subresult in self.result]
        num_post_tup, num_oot_tup = zip(*temp)
        num_post, num_oot = list(set(num_post_tup)), list(set(num_oot_tup))

        if len(num_post) > 1 or len(num_oot) > 1:
            raise Exception('Number of posts or OOT posts mismatch')
        if len(num_post) == 0 or len(num_oot) == 0:
            return 0, 0
        else:
            return num_post[0], num_oot[0]

    @lazyproperty
    def min_sup(self):
        """Return the minimum support value of random variable X.

        X is a hypergeometric random variable associated with this event.
        """
        return max(self.N - self.M + self.n, 0)

    @lazyproperty
    def max_sup(self):
        """Return the maximum support value of random variable X.

        X is a hypergeometric random variable associated with this event.
        """
        return min(self.N, self.n)

    @lazyproperty
    def baseline(self):
        """Return the baseline performance vector.

        The baseline is obtaining OOT posts by chance. Thus, the baseline
        performance vector is the probability mass function of a hypergeometric
        random variable denoting the number of OOT posts in the top N list.
        The k-th element represents the probability of getting k OOT posts in
        the top N list.
        """
        rv = hypergeom(self.M, self.n, self.N)
        k = np.arange(self.min_sup, self.max_sup+1)
        return rv.pmf(k)

    @lazyproperty
    def performance(self):
        """Return the evaluation result in a performance vector."""
        num_expr = len(self.result)
        if num_expr == 0:
            raise Exception('No experiment error')
        top_oot_nums = [sum(is_oot for _, is_oot in subresult[:self.N])
                        for subresult in self.result]

        length = self.max_sup - self.min_sup + 1
        res = np.zeros(length)
        count = Counter(top_oot_nums)
        for k in range(length):
            res[k] = count[k] / num_expr
        return res
