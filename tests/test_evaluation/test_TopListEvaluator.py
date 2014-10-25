from nose.tools import assert_equal, assert_greater

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
