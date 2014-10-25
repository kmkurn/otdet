from numpy.testing import assert_almost_equal
import numpy as np
from unittest.mock import patch

from otdet.detector import OOTDetector


class TestClustDist:
    def setUp(self):
        self.detector = OOTDetector(['test1.txt', 'test2.txt'])

    def test_univariate(self):
        X = np.array([[2], [-1], [3]])
        with patch('otdet.detector.OOTDetector.design_matrix', X):
            expected = np.array([4/3, 7/3, 5/3])
            result = self.detector.clust_dist()
            assert_almost_equal(result, expected)

    def test_multivariate(self):
        X = np.array([[2, 1, 0], [-1, 3, 4], [2, -2, 1]])
        with patch('otdet.detector.OOTDetector.design_matrix', X):
            expected = np.array([13/3, 20/3, 5])
            result = self.detector.clust_dist(metric='cityblock')
            assert_almost_equal(result, expected)


class TestMeanComp:
    def setUp(self):
        self.detector = OOTDetector(['test1.txt', 'test2.txt'])

    def test_univariate(self):
        X = np.array([[2], [-1], [3]])
        with patch('otdet.detector.OOTDetector.design_matrix', X):
            expected = np.array([1, 7/2, 5/2])
            result = self.detector.mean_comp()
            assert_almost_equal(result, expected)

    def test_multivariate(self):
        X = np.array([[2, 1, 0], [-1, 3, 4], [2, -2, 1]])
        with patch('otdet.detector.OOTDetector.design_matrix', X):
            expected = np.array([9/2, 10, 13/2])
            result = self.detector.mean_comp(metric='cityblock')
            assert_almost_equal(result, expected)


class TestTxtCompDist:
    def setUp(self):
        self.detector = OOTDetector(['test1.txt', 'test2.txt'])

    def test_univariate(self):
        sample_contents = [
            'ani ani\n',
            'ani ani ani\n',
            'ani ani ani ani ani\n'
        ]
        with patch('otdet.detector.OOTDetector.contents', sample_contents):
            expected = np.array([6, 4, 0])
            result = self.detector.txt_comp_dist()
            assert_almost_equal(result, expected)

    def test_multivariate(self):
        sample_contents = [
            'ani budi budi cika\n',
            'ani budi\n',
            'cika cika\n'
        ]
        with patch('otdet.detector.OOTDetector.contents', sample_contents):
            expected = np.sqrt(np.array([2, 10, 14]))
            result = self.detector.txt_comp_dist()
            assert_almost_equal(result, expected)
