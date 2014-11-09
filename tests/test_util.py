from otdet.util import pick

from nose.tools import assert_equal, assert_true, raises


class TestPick():
    def setUp(self):
        self.filenames = ['a-4.txt', 'b-2.txt', 'c-3.txt', 'd-1.txt',
                          'e-0.txt']

    def test_all_sequential(self):
        expected = ['e-0.txt', 'd-1.txt', 'b-2.txt', 'c-3.txt', 'a-4.txt']
        result = pick(self.filenames, k=100, randomized=False)
        assert_equal(result, expected)

    def test_k_sequential(self):
        expected = ['e-0.txt', 'd-1.txt', 'b-2.txt']
        result = pick(self.filenames, k=3, randomized=False)
        assert_equal(result, expected)

    def test_all_random(self):
        result = pick(self.filenames, k=100)
        assert_equal(sorted(self.filenames), sorted(result))

    def test_k_random(self):
        result = pick(self.filenames, k=3)
        assert_equal(len(result), 3)
        for r in result:
            assert_true(r in self.filenames)

    @raises(Exception)
    def test_negative_k(self):
        pick(self.filenames, k=-2)
