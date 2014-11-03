from nose.tools import raises
from numpy.testing import assert_almost_equal
import numpy as np

from otdet.evaluation import TopListEvaluator


class TestBaseline:
    def setUp(self):
        self.sample_result = [(5.0, True), (4.0, False), (3.0, True),
                              (2.0, False), (1.0, False)]
        self.evaluator = TopListEvaluator(M=5, n=2)

    def test_normal_case(self):
        self.evaluator.N = 3
        expected = np.array([0.1, 0.6, 0.3])
        assert_almost_equal(self.evaluator.baseline, expected)

    def test_top_few_list(self):
        self.evaluator.N = 1
        expected = np.array([0.6, 0.4, 0.0])
        assert_almost_equal(self.evaluator.baseline, expected)

    def test_top_many_list(self):
        self.evaluator.N = 4
        expected = np.array([0.0, 0.4, 0.6])
        assert_almost_equal(self.evaluator.baseline, expected)


class TestGetPerformance:
    def setUp(self):
        sample_result1 = list(zip(range(5, 0, -1),
                                  [True, False, False, True, False]))
        sample_result2 = list(zip(range(5, 0, -1),
                                  [True, False, True, False, False]))
        sample_result3 = list(zip(range(5, 0, -1),
                                  [False, True, True, False, False]))
        sample_result4 = list(zip(range(5, 0, -1),
                                  [False, False, False, True, True]))
        self.result_list = [sample_result1, sample_result2,
                            sample_result3, sample_result4]
        self.evaluator = TopListEvaluator(M=5, n=2)

    def test_normal_case(self):
        self.evaluator.N = 3
        expected = np.array([0.25, 0.25, 0.50])
        result = self.evaluator.get_performance(self.result_list)
        assert_almost_equal(result, expected)

    @raises(Exception)
    def test_no_experiment(self):
        TopListEvaluator().get_performance([])

    @raises(Exception)
    def test_total_post_number_mismatch(self):
        sample_result1 = list(zip(range(3, 0, -1), [True, False, False]))
        sample_result2 = list(zip(range(2, 0, -1), [True, False]))
        TopListEvaluator(M=3, n=1).get_performance([sample_result1,
                                                    sample_result2])

    @raises(Exception)
    def test_oot_post_number_mismatch(self):
        sample_result1 = list(zip(range(3, 0, -1), [True, False, False]))
        sample_result2 = list(zip(range(3, 0, -1), [True, False, True]))
        TopListEvaluator(M=3, n=1).get_performance([sample_result1,
                                                    sample_result2])

    @raises(Exception)
    def test_oot_post_number_mismatch2(self):
        sample_result1 = list(zip(range(3, 0, -1), [True, False, False]))
        sample_result2 = list(zip(range(3, 0, -1), [True, False, False]))
        TopListEvaluator(M=3, n=2).get_performance([sample_result1,
                                                    sample_result2])
