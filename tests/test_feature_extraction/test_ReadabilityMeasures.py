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
    @patch.object(ReadabilityMeasures, 'smog', return_value=1)
    @patch.object(ReadabilityMeasures, 'ari', return_value=2)
    @patch.object(ReadabilityMeasures, 'lix', return_value=3)
    def test_partial_measures(self, mock_lix, mock_ari, mock_smog):
        extractor = ReadabilityMeasures(measures=['smog', 'ari', 'lix'])
        expected = np.arange(1, 4)
        result = extractor._to_vector([])
        assert_almost_equal(result, expected)


class TestTotalSents:
    def test_default(self):
        tokenized_content = [
            ['ani', 'budi', 'cika'],
            ['ani', 'cika'],
            ['budi']
        ]
        assert_equal(ReadabilityMeasures.total_sents(tokenized_content), 3)

    def test_no_sentence(self):
        tokenized_content = []
        assert_equal(ReadabilityMeasures.total_sents(tokenized_content), 0)


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
    @patch.object(ReadabilityMeasures, 'num_syllables')
    def test_default(self, mock_num_syllables):
        mock_num_syllables.side_effect = [1, 2, 3, 3, 3, 2, 1]
        tokenized_content = [['aa', 'aaa', 'aa'], ['aa', 'aaaaa'],
                             ['a', 'aaa']]
        result = ReadabilityMeasures.total_sylls(tokenized_content)
        assert_almost_equal(result, 15)


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
    @patch.object(ReadabilityMeasures, 'total_sents')
    @patch.object(ReadabilityMeasures, 'total_words')
    @patch.object(ReadabilityMeasures, 'total_sylls')
    def test_default(self, mock_total_sylls, mock_total_words,
                     mock_total_sents):
        mock_total_sylls.return_value = 50
        mock_total_words.return_value = 30
        mock_total_sents.return_value = 5
        result = ReadabilityMeasures.fleschease([])
        assert_almost_equal(result, 59.745)

    @patch.object(ReadabilityMeasures, 'total_words')
    def test_zero_words(self, mock_total_words):
        mock_total_words.return_value = 0
        result = ReadabilityMeasures.fleschease([])
        assert_almost_equal(result, ReadabilityMeasures.INF)

    @patch.object(ReadabilityMeasures, 'total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.fleschease([])
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestFleschgrade:
    @patch.object(ReadabilityMeasures, 'total_sents')
    @patch.object(ReadabilityMeasures, 'total_words')
    @patch.object(ReadabilityMeasures, 'total_sylls')
    def test_default(self, mock_total_sylls, mock_total_words,
                     mock_total_sents):
        mock_total_sylls.return_value = 50
        mock_total_words.return_value = 30
        mock_total_sents.return_value = 5
        result = ReadabilityMeasures.fleschgrade([])
        assert_almost_equal(result, 6.41666667)

    @patch.object(ReadabilityMeasures, 'total_words')
    def test_zero_words(self, mock_total_words):
        mock_total_words.return_value = 0
        result = ReadabilityMeasures.fleschgrade([])
        assert_almost_equal(result, ReadabilityMeasures.INF)

    @patch.object(ReadabilityMeasures, 'total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.fleschgrade([])
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestFogindex:
    @patch.object(ReadabilityMeasures, 'total_sents')
    @patch.object(ReadabilityMeasures, 'total_words')
    @patch.object(ReadabilityMeasures, 'num_syllables')
    def test_default(self, mock_num_syllables, mock_total_words,
                     mock_total_sents):
        mock_num_syllables.side_effect = [8, 2, 3, 1, 5, 6, 7]
        mock_total_words.return_value = 30
        mock_total_sents.return_value = 5
        tokenized_content = [['a', 'b'], ['c', 'd', 'e', 'f', 'g']]
        result = ReadabilityMeasures.fogindex(tokenized_content)
        assert_almost_equal(result, 22.66666667)

    @patch.object(ReadabilityMeasures, 'total_words')
    def test_zero_words(self, mock_total_words):
        mock_total_words.return_value = 0
        result = ReadabilityMeasures.fogindex([])
        assert_almost_equal(result, ReadabilityMeasures.INF)

    @patch.object(ReadabilityMeasures, 'total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.fogindex([])
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestColemanliau:
    @patch.object(ReadabilityMeasures, 'total_sents')
    @patch.object(ReadabilityMeasures, 'total_words')
    @patch.object(ReadabilityMeasures, 'total_chars')
    def test_default(self, mock_total_chars, mock_total_words,
                     mock_total_sents):
        mock_total_chars.return_value = 100
        mock_total_words.return_value = 30
        mock_total_sents.return_value = 5
        result = ReadabilityMeasures.colemanliau([])
        assert_almost_equal(result, 3.83283333)

    @patch.object(ReadabilityMeasures, 'total_words')
    def test_zero_words(self, mock_total_words):
        mock_total_words.return_value = 0
        result = ReadabilityMeasures.colemanliau([])
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestAri:
    @patch.object(ReadabilityMeasures, 'total_sents')
    @patch.object(ReadabilityMeasures, 'total_words')
    @patch.object(ReadabilityMeasures, 'total_chars')
    def test_default(self, mock_total_chars, mock_total_words,
                     mock_total_sents):
        mock_total_chars.return_value = 100
        mock_total_words.return_value = 30
        mock_total_sents.return_value = 5
        result = ReadabilityMeasures.ari([])
        assert_almost_equal(result, -2.72999999)

    @patch.object(ReadabilityMeasures, 'total_words')
    def test_zero_words(self, mock_total_words):
        mock_total_words.return_value = 0
        result = ReadabilityMeasures.ari([])
        assert_almost_equal(result, ReadabilityMeasures.INF)

    @patch.object(ReadabilityMeasures, 'total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.ari([])
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestLix:
    @patch.object(ReadabilityMeasures, 'total_sents')
    @patch.object(ReadabilityMeasures, 'total_words')
    def test_default(self, mock_total_words, mock_total_sents):
        mock_total_words.return_value = 30
        mock_total_sents.return_value = 5
        tokenized_content = [
            ['a', 'aaaaaa'],
            ['aa', 'aaa', 'aaaaaaaa', 'aa', 'aaaaaaa']
        ]
        result = ReadabilityMeasures.lix(tokenized_content)
        assert_almost_equal(result, 16)

    @patch.object(ReadabilityMeasures, 'total_words')
    def test_zero_words(self, mock_total_words):
        mock_total_words.return_value = 0
        result = ReadabilityMeasures.lix([])
        assert_almost_equal(result, ReadabilityMeasures.INF)

    @patch.object(ReadabilityMeasures, 'total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.lix([])
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestSmog:
    @patch.object(ReadabilityMeasures, 'total_sents')
    @patch.object(ReadabilityMeasures, 'num_syllables')
    def test_default(self, mock_num_syllables, mock_total_sents):
        mock_num_syllables.side_effect = [8, 2, 3, 1, 5, 6, 7]
        mock_total_sents.return_value = 5
        tokenized_content = [['a', 'b'], ['c', 'd', 'e', 'f', 'g']]
        result = ReadabilityMeasures.smog(tokenized_content)
        assert_almost_equal(result, 8.47722557)

    @patch.object(ReadabilityMeasures, 'total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.smog([])
        assert_almost_equal(result, ReadabilityMeasures.INF)


d = {
    'a': [['a1', 'b', 'c'], ['a', 'b2', 'c1']],
    'b': [['a2', 'b1']],
    'c': [['a', 'b', 'c'], ['a']],
    'ab': [['a', 'b1'], ['b2', 'c3']]
}


@patch.object(ReadabilityMeasures, 'd', d)
class TestNumSyllables:
    def test_word_exist_in_corpus(self):
        assert_almost_equal(ReadabilityMeasures.num_syllables('a'), 1.5)

    def test_word_exist_in_corpus2(self):
        assert_almost_equal(ReadabilityMeasures.num_syllables('c'), 0)

    @patch.object(ReadabilityMeasures, 'avg_syllables')
    def test_word_not_exist_in_corpus(self, mock_avg_syllables):
        mock_avg_syllables.return_value = 5
        assert_almost_equal(ReadabilityMeasures.num_syllables('f'), 5)


@patch.object(ReadabilityMeasures, 'd', d)
class TestAvgSyllables:
    def test_word_len_exist_in_corpus(self):
        assert_almost_equal(ReadabilityMeasures.avg_syllables(1), 7/6)

    def test_word_len_not_exist_in_corpus(self):
        assert_almost_equal(ReadabilityMeasures.avg_syllables(3), 5/4)
