import util

from nose.tools import assert_equal


class TestPick():
    def check(self, filenames, expected, k, randomized):
        result = util.pick(filenames, k, randomized)
        assert_equal(result, expected)

    def test_all_sequential(self):
        filenames = ['a-4.txt', 'b-2.txt', 'c-3.txt', 'd-1.txt', 'e-0.txt']
        expected = ['e-0.txt', 'd-1.txt', 'b-2.txt', 'c-3.txt', 'a-4.txt']
        self.check(filenames, expected, k=None, randomized=False)
