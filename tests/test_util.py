from otdet.util import pick

from nose.tools import assert_equal, assert_true, raises


class TestPick():
    def test_all_sequential(self):
        filenames = ['a-4.txt', 'b-2.txt', 'c-3.txt', 'd-1.txt', 'e-0.txt']
        expected = ['e-0.txt', 'd-1.txt', 'b-2.txt', 'c-3.txt', 'a-4.txt']
        result = pick(filenames, randomized=False)
        assert_equal(result, expected)

    def test_k_sequential(self):
        filenames = ['a-4.txt', 'b-2.txt', 'c-3.txt', 'd-1.txt', 'e-0.txt']
        expected = ['e-0.txt', 'd-1.txt', 'b-2.txt']
        result = pick(filenames, k=3, randomized=False)
        assert_equal(result, expected)

    def test_all_random(self):
        filenames = ['a-4.txt', 'b-2.txt', 'c-3.txt', 'd-1.txt', 'e-0.txt']
        result = pick(filenames)
        assert_equal(sorted(filenames), sorted(result))

    def test_k_random(self):
        filenames = ['a-4.txt', 'b-2.txt', 'c-3.txt', 'd-1.txt', 'e-0.txt']
        result = pick(filenames, k=3)
        for r in result:
            assert_true(r in filenames)

    @raises(ValueError)
    def test_negative_k(self):
        pick([], k=-2)
