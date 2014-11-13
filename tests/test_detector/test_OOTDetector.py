from nose.tools import assert_true
from numpy.testing import assert_almost_equal
import numpy as np
from unittest.mock import patch

from otdet.detector import OOTDetector
from otdet.feature_extraction import CountVectorizerWrapper, \
    ReadabilityMeasures


class TestInit:
    def test_default(self):
        detector = OOTDetector()
        assert_true(hasattr(detector, 'extractor'))
        assert_true(isinstance(detector.extractor, CountVectorizerWrapper))

    def test_custom_extractor(self):
        detector = OOTDetector(extractor=ReadabilityMeasures())
        assert_true(isinstance(detector.extractor, ReadabilityMeasures))


class TestDesignMatrix:
    @patch.object(CountVectorizerWrapper, 'fit_transform')
    def test_default(self, mock_fit_transform):
        expected = np.array([[1, 2, 2], [2, 1, 2]])
        mock_fit_transform.return_value = expected
        detector = OOTDetector()
        documents = ['a b c. c b.', 'b c. a a c.']
        result = detector.design_matrix(documents)
        assert_almost_equal(result, expected)
        mock_fit_transform.assert_called_with(documents)


@patch.object(OOTDetector, 'design_matrix')
class TestClustDist:
    def setUp(self):
        self.detector = OOTDetector()
        self.documents = ['a b c. c b.', 'b c. a a c.']

    def test_univariate(self, mock_design_matrix):
        X = np.array([[2], [-1], [3]])
        mock_design_matrix.return_value = X
        expected = np.array([4/3, 7/3, 5/3])
        result = self.detector.clust_dist(self.documents)
        assert_almost_equal(result, expected)
        mock_design_matrix.assert_called_with(self.documents)

    def test_multivariate(self, mock_design_matrix):
        X = np.array([[2, 1, 0], [-1, 3, 4], [2, -2, 1]])
        mock_design_matrix.return_value = X
        expected = np.array([13/3, 20/3, 5])
        result = self.detector.clust_dist(self.documents,
                                          metric='cityblock')
        assert_almost_equal(result, expected)
        mock_design_matrix.assert_called_with(self.documents)


@patch.object(OOTDetector, 'design_matrix')
class TestMeanComp:
    def setUp(self):
        self.detector = OOTDetector()
        self.documents = ['a b c. c b.', 'b c. a a c.']

    def test_univariate(self, mock_design_matrix):
        X = np.array([[2], [-1], [3]])
        mock_design_matrix.return_value = X
        expected = np.array([1, 7/2, 5/2])
        result = self.detector.mean_comp(self.documents)
        assert_almost_equal(result, expected)
        mock_design_matrix.assert_called_with(self.documents)

    def test_multivariate(self, mock_design_matrix):
        X = np.array([[2, 1, 0], [-1, 3, 4], [2, -2, 1]])
        mock_design_matrix.return_value = X
        expected = np.array([9/2, 10, 13/2])
        result = self.detector.mean_comp(self.documents,
                                         metric='cityblock')
        assert_almost_equal(result, expected)
        mock_design_matrix.assert_called_with(self.documents)


class TestTxtCompDist:
    def setUp(self):
        self.detector = OOTDetector()

    def test_univariate(self):
        sample_contents = [
            'ani ani\n',
            'ani ani ani\n',
            'ani ani ani ani ani\n'
        ]
        expected = np.array([6, 4, 0])
        result = self.detector.txt_comp_dist(sample_contents)
        assert_almost_equal(result, expected)

    def test_multivariate(self):
        sample_contents = [
            'ani budi budi cika\n',
            'ani budi\n',
            'cika cika\n'
        ]
        expected = np.sqrt(np.array([2, 10, 14]))
        result = self.detector.txt_comp_dist(sample_contents)
        assert_almost_equal(result, expected)
