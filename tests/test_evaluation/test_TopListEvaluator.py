from nose.tools import assert_equal, assert_greater, raises
from numpy.testing import assert_almost_equal
import numpy as np

from otdet.evaluation import TopListEvaluator


class TestAddResult:
    def setUp(self):
        self.sample_result = [(5.0, True), (4.0, False), (3.0, True),
                              (2.0, False), (1.0, False)]
        self.M = len(self.sample_result)
        self.n = sum(elm[1] for elm in self.sample_result)

    def test_normal_result(self):
        N = 2
        k = sum(elm[1] for elm in self.sample_result[:N])
        evaluator = TopListEvaluator(N)
        evaluator.add_result(self.sample_result)
        assert_equal(evaluator._M, self.M)
        assert_equal(evaluator._n, self.n)
        assert_equal(evaluator._numexpr, 1)
        assert_equal(evaluator._freq[k], 1)

    def test_short_result(self):
        N = 10
        k = sum(elm[1] for elm in self.sample_result[:N])
        evaluator = TopListEvaluator(N)
        evaluator.add_result(self.sample_result)
        assert_equal(evaluator._M, self.M)
        assert_equal(evaluator._n, self.n)
        assert_equal(evaluator._numexpr, 1)
        assert_equal(evaluator._freq[k], 1)

    def test_called_twice(self):
        N = 2
        evaluator = TopListEvaluator(N)
        evaluator.add_result(self.sample_result)
        evaluator.add_result(self.sample_result)
        assert_equal(evaluator._numexpr, 2)
        assert_greater(len(evaluator._result_list), 0)
        assert_equal(evaluator._result_list[0], self.sample_result)


class TestBaseline:
    def setUp(self):
        self.sample_result = [(5.0, True), (4.0, False), (3.0, True),
                              (2.0, False), (1.0, False)]

    def test_normal_case(self):
        N = 3
        evaluator = TopListEvaluator(N)
        evaluator.add_result(self.sample_result)
        expected = np.array([0.1, 0.6, 0.3])
        assert_almost_equal(evaluator.baseline, expected)

    def test_top_few_list(self):
        N = 1
        evaluator = TopListEvaluator(N)
        evaluator.add_result(self.sample_result)
        expected = np.array([0.6, 0.4, 0.0])
        assert_almost_equal(evaluator.baseline, expected)

    def test_top_many_list(self):
        N = 4
        evaluator = TopListEvaluator(N)
        evaluator.add_result(self.sample_result)
        expected = np.array([0.0, 0.4, 0.6])
        assert_almost_equal(evaluator.baseline, expected)


class TestGetPerformance:
    def setUp(self):
        sample_result1 = zip(range(5, 0, -1),
                             [True, False, False, True, False])
        sample_result2 = zip(range(5, 0, -1),
                             [True, False, True, False, False])
        sample_result3 = zip(range(5, 0, -1),
                             [False, True, True, False, False])
        sample_result4 = zip(range(5, 0, -1),
                             [False, False, False, True, True])
        self.result_list = [sample_result1, sample_result2,
                            sample_result3, sample_result4]

    def test_normal_case(self):
        N = 3
        evaluator = TopListEvaluator(N)
        for sample_result in self.result_list:
            evaluator.add_result(sample_result)
        expected = np.array([0.25, 0.25, 0.50])
        assert_almost_equal(evaluator.get_performance, expected)

    @raises(Exception)
    def test_no_experiment(self):
        TopListEvaluator().get_performance
