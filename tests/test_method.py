from numpy.testing import assert_almost_equal
import numpy as np

import method


class TestClustDist():
    def test_univariate(self):
        X = np.array([[2], [-1], [3]])
        expected = np.array([4/3, 7/3, 5/3])
        result = method.clust_dist(X)
        assert_almost_equal(result, expected)

    def test_multivariate(self):
        X = np.array([[2, 1, 0], [-1, 3, 4], [2, -2, 1]])
        expected = np.array([13/3, 20/3, 5])
        result = method.clust_dist(X, metric='cityblock')
        assert_almost_equal(result, expected)


class TestMeanComp():
    def test_univariate(self):
        X = np.array([[2], [-1], [3]])
        expected = np.array([1, 7/2, 5/2])
        result = method.mean_comp(X)
        assert_almost_equal(result, expected)

    def test_multivariate(self):
        X = np.array([[2, 1, 0], [-1, 3, 4], [2, -2, 1]])
        expected = np.array([9/2, 10, 13/2])
        result = method.mean_comp(X, metric='cityblock')
        assert_almost_equal(result, expected)
