from nose.tools import raises, assert_equal
from numpy.testing import assert_almost_equal
import numpy as np

from otdet.evaluation import TopListEvaluator


class TestInit:
    def test_default(self):
        sample_result = [
            [(5, True), (4, False), (3, True), (2, False), (1, False)],
            [(5, False), (4, True), (3, False), (2, True), (1, False)]
        ]
        evaluator = TopListEvaluator(sample_result)
        assert_equal(evaluator._M, 5)
        assert_equal(evaluator._n, 2)
        assert_equal(evaluator.N, 1)

    def test_pick_many_top_list(self):
        sample_result = [
            [(5, True), (4, False), (3, True), (2, False), (1, False)],
            [(5, False), (4, True), (3, False), (2, True), (1, False)]
        ]
        evaluator = TopListEvaluator(sample_result, N=3)
        assert_equal(evaluator.N, 3)

    @raises(Exception)
    def test_num_oot_mismatch(self):
        sample_result = [
            [(5, True), (4, True), (3, True), (2, False), (1, False)],
            [(5, False), (4, True), (3, False), (2, True), (1, False)]
        ]
        TopListEvaluator(sample_result)

    @raises(Exception)
    def test_num_post_mismatch(self):
        sample_result = [
            [(5, True), (4, True), (3, True), (2, False)],
            [(5, False), (4, True), (3, False), (2, True), (1, False)]
        ]
        TopListEvaluator(sample_result)

    @raises(Exception)
    def test_pick_negative(self):
        sample_result = [
            [(5, True), (4, False), (3, True), (2, False), (1, False)],
            [(5, False), (4, True), (3, False), (2, True), (1, False)]
        ]
        TopListEvaluator(sample_result, N=-1)


class TestBaseline:
    def setUp(self):
        sample_result = [
            [(5, True), (4, False), (3, True), (2, False), (1, False)],
            [(5, False), (4, True), (3, False), (2, True), (1, False)]
        ]
        self.evaluator = TopListEvaluator(sample_result, N=3)

    def test_normal_case(self):
        expected = np.array([0.1, 0.6, 0.3])  # 0 <= k <= 2
        assert_almost_equal(self.evaluator.baseline, expected)

    def test_top_few_list(self):
        self.evaluator.N = 1
        expected = np.array([0.6, 0.4])  # 0 <= k <= 1
        assert_almost_equal(self.evaluator.baseline, expected)

    def test_top_many_list(self):
        self.evaluator.N = 4
        expected = np.array([0.4, 0.6])  # 1 <= k <= 2
        assert_almost_equal(self.evaluator.baseline, expected)


class TestPerformance:
    def setUp(self):
        sample_result = [
            [(5, True), (4, False), (3, False), (2, True), (1, False)],
            [(5, True), (4, False), (3, True), (2, False), (1, False)],
            [(5, False), (4, True), (3, True), (2, False), (1, False)],
            [(5, False), (4, False), (3, False), (2, True), (1, True)]
        ]
        self.evaluator = TopListEvaluator(sample_result, N=3)

    def test_normal_case(self):
        expected = np.array([0.25, 0.25, 0.50])  # 0 <= k <= 2
        result = self.evaluator.performance
        assert_almost_equal(result, expected)
