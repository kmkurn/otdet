from unittest import TestCase
from unittest.mock import patch

from nose.tools import assert_equal
import numpy as np
from numpy.testing import assert_almost_equal

from otdet.feature_extraction import ReadabilityMeasures


class TestTokenizeContent(TestCase):
    def setUp(self):
        self.content = 'Ani Budi Cika. Ani Cika.\nBudi.\n'
        p1 = patch('otdet.feature_extraction.sent_tokenize')
        p2 = patch('otdet.feature_extraction.word_tokenize')
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        self.mock_sent_tokenize = p1.start()
        self.mock_word_tokenize = p2.start()

    def test_default(self):
        self.mock_sent_tokenize.return_value = range(3)
        self.mock_word_tokenize.side_effect = [
            ['Ani', 'Budi', 'Cika', '.'],
            ['Ani', 'Cika', '.'],
            ['Budi', '.']
        ]
        extractor = ReadabilityMeasures()
        expected = [
            ['Ani', 'Budi', 'Cika'],
            ['Ani', 'Cika'],
            ['Budi']
        ]
        assert_equal(extractor._tokenize_content(self.content), expected)
        self.mock_sent_tokenize.assert_called_with(self.content)

    def test_no_remove_punct(self):
        self.mock_sent_tokenize.return_value = range(3)
        self.mock_word_tokenize.side_effect = [
            ['Ani', 'Budi', 'Cika', '.'],
            ['Ani', 'Cika', '.'],
            ['Budi', '.']
        ]
        extractor = ReadabilityMeasures(remove_punct=False)
        expected = [
            ['Ani', 'Budi', 'Cika', '.'],
            ['Ani', 'Cika', '.'],
            ['Budi', '.']
        ]
        assert_equal(extractor._tokenize_content(self.content), expected)

    def test_all_punct(self):
        self.mock_sent_tokenize.return_value = range(3)
        self.mock_word_tokenize.side_effect = [
            ['.', '.', '.', '.'],
            ['Ani', '!', '?'],
            ['Cika', '.']
        ]
        extractor = ReadabilityMeasures()
        expected = [['Ani'], ['Cika']]
        assert_equal(extractor._tokenize_content(self.content), expected)


class TestToVector:
    def setUp(self):
        self.tokenized_content = [
            ['under', 'pressure'],
            ['she', 'is', 'gregarious', 'and', 'gorgeous']
        ]

    def test_default(self):
        extractor = ReadabilityMeasures()
        expected = np.array([46.16821429, 7.68928571, 17.78571429,
                             17.01485714, 6.5614286, 46.3571429, 6.8729833])
        result = extractor._to_vector(self.tokenized_content)
        assert_almost_equal(result, expected)

    def test_partial_measures(self):
        extractor = ReadabilityMeasures(measures=['smog', 'ari', 'lix'])
        expected = np.array([6.8729833, 6.5614286, 46.3571429])
        result = extractor._to_vector(self.tokenized_content)
        assert_almost_equal(result, expected)


class TestTotalSents:
    def test_default(self):
        tokenized_content = [
            ['ani', 'budi', 'cika'],
            ['ani', 'cika'],
            ['budi']
        ]
        extractor = ReadabilityMeasures()
        assert_equal(extractor.total_sents(tokenized_content), 3)


class TestTotalWords:
    def test_default(self):
        tokenized_content = [
            ['ani', 'budi', 'cika'],
            ['ani', 'cika'],
            ['budi']
        ]
        extractor = ReadabilityMeasures()
        assert_equal(extractor.total_words(tokenized_content), 6)


class TestTotalSylls:
    def test_default(self):
        tokenized_content = [
            ['love', 'of', 'my', 'life'],
            ['crazy', 'little', 'thing', 'called', 'love'],
            ['under', 'pressure']
        ]
        result = ReadabilityMeasures.total_sylls(tokenized_content)
        assert_almost_equal(result, 15)

    def test_unknown_words(self):
        tokenized_content = [
            ['ani', 'budi', 'cika'],
            ['ani', 'cika'],
            ['budi']
        ]
        result = ReadabilityMeasures.total_sylls(tokenized_content)
        expected = 7.838407863890021
        assert_almost_equal(result, expected)


class TestTotalChars:
    def test_default(self):
        tokenized_content = [
            ['ani', 'budi', 'cika'],
            ['ani', 'cika'],
            ['budi']
        ]
        extractor = ReadabilityMeasures()
        assert_equal(extractor.total_chars(tokenized_content), 22)


class TestFleschease:
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sents')
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_words')
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sylls')
    def test_default(self, mock_total_sylls, mock_total_words,
                     mock_total_sents):
        mock_total_sylls.return_value = 50
        mock_total_words.return_value = 30
        mock_total_sents.return_value = 5
        result = ReadabilityMeasures.fleschease([])
        assert_almost_equal(result, 59.745)

    @patch('otdet.feature_extraction.ReadabilityMeasures.total_words')
    def test_zero_words(self, mock_total_words):
        mock_total_words.return_value = 0
        result = ReadabilityMeasures.fleschease([])
        assert_almost_equal(result, ReadabilityMeasures.INF)

    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.fleschease([])
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestFleschgrade:
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sents')
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_words')
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sylls')
    def test_default(self, mock_total_sylls, mock_total_words,
                     mock_total_sents):
        mock_total_sylls.return_value = 50
        mock_total_words.return_value = 30
        mock_total_sents.return_value = 5
        result = ReadabilityMeasures.fleschgrade([])
        assert_almost_equal(result, 6.41666667)

    @patch('otdet.feature_extraction.ReadabilityMeasures.total_words')
    def test_zero_words(self, mock_total_words):
        mock_total_words.return_value = 0
        result = ReadabilityMeasures.fleschgrade([])
        assert_almost_equal(result, ReadabilityMeasures.INF)

    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.fleschgrade([])
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestFogindex:
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sents')
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_words')
    @patch('otdet.feature_extraction.ReadabilityMeasures.num_syllables')
    def test_default(self, mock_numsyllables, mock_total_words,
                     mock_total_sents):
        mock_numsyllables.side_effect = [8, 2, 3, 1, 5, 6, 7]
        mock_total_words.return_value = 30
        mock_total_sents.return_value = 5
        tokenized_content = [['a', 'b'], ['c', 'd', 'e', 'f', 'g']]
        result = ReadabilityMeasures.fogindex(tokenized_content)
        assert_almost_equal(result, 22.66666667)

    @patch('otdet.feature_extraction.ReadabilityMeasures.total_words')
    def test_zero_words(self, mock_total_words):
        mock_total_words.return_value = 0
        result = ReadabilityMeasures.fogindex([])
        assert_almost_equal(result, ReadabilityMeasures.INF)

    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.fogindex([])
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestColemanliau:
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sents')
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_words')
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_chars')
    def test_default(self, mock_total_chars, mock_total_words,
                     mock_total_sents):
        mock_total_chars.return_value = 100
        mock_total_words.return_value = 30
        mock_total_sents.return_value = 5
        result = ReadabilityMeasures.colemanliau([])
        assert_almost_equal(result, 3.83283333)

    @patch('otdet.feature_extraction.ReadabilityMeasures.total_words')
    def test_zero_words(self, mock_total_words):
        mock_total_words.return_value = 0
        result = ReadabilityMeasures.colemanliau([])
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestAri:
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sents')
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_words')
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_chars')
    def test_default(self, mock_total_chars, mock_total_words,
                     mock_total_sents):
        mock_total_chars.return_value = 100
        mock_total_words.return_value = 30
        mock_total_sents.return_value = 5
        result = ReadabilityMeasures.ari([])
        assert_almost_equal(result, -2.72999999)

    @patch('otdet.feature_extraction.ReadabilityMeasures.total_words')
    def test_zero_words(self, mock_total_words):
        mock_total_words.return_value = 0
        result = ReadabilityMeasures.ari([])
        assert_almost_equal(result, ReadabilityMeasures.INF)

    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.ari([])
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestLix:
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sents')
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_words')
    def test_default(self, mock_total_words, mock_total_sents):
        mock_total_words.return_value = 30
        mock_total_sents.return_value = 5
        tokenized_content = [
            ['a', 'aaaaaa'],
            ['aa', 'aaa', 'aaaaaaaa', 'aa', 'aaaaaaa']
        ]
        result = ReadabilityMeasures.lix(tokenized_content)
        assert_almost_equal(result, 16)

    @patch('otdet.feature_extraction.ReadabilityMeasures.total_words')
    def test_zero_words(self, mock_total_words):
        mock_total_words.return_value = 0
        result = ReadabilityMeasures.lix([])
        assert_almost_equal(result, ReadabilityMeasures.INF)

    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.lix([])
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestSmog:
    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sents')
    @patch('otdet.feature_extraction.ReadabilityMeasures.num_syllables')
    def test_default(self, mock_num_syllables, mock_total_sents):
        mock_num_syllables.side_effect = [8, 2, 3, 1, 5, 6, 7]
        mock_total_sents.return_value = 5
        tokenized_content = [['a', 'b'], ['c', 'd', 'e', 'f', 'g']]
        result = ReadabilityMeasures.smog(tokenized_content)
        assert_almost_equal(result, 8.47722557)

    @patch('otdet.feature_extraction.ReadabilityMeasures.total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.smog([])
        assert_almost_equal(result, ReadabilityMeasures.INF)
