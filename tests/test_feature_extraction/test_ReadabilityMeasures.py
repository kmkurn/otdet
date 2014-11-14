from unittest import TestCase
from unittest.mock import call, patch, Mock, MagicMock

from nose.tools import assert_equal
import numpy as np
from numpy.testing import assert_almost_equal

from otdet.feature_extraction import ReadabilityMeasures, TokenizedContent


class TestFitTransform:
    def setUp(self):
        self.documents = ['First document.', 'Second.\nDocument.\n']

    @patch.object(ReadabilityMeasures, 'transform')
    def test_default(self, mock_transform):
        expected = np.array([np.array([1, 2]), np.array([3, 4])])
        mock_transform.return_value = expected
        extractor = ReadabilityMeasures()
        assert_almost_equal(extractor.fit_transform(self.documents), expected)
        mock_transform.assert_called_with(self.documents)


@patch.object(ReadabilityMeasures, '_to_vector')
@patch.object(ReadabilityMeasures, '_tokenize_content')
class TestTransform:
    def setUp(self):
        self.documents = ['aa aaab. aab. a.', 'aa.\n', 'aa.\naaab.\n\naa!!']

    def test_default(self, mock_tokenize_content, mock_to_vector):
        expected = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        mock_tokenize_content.side_effect = [
            [[['aa'], ['aaab']], [['aab']], [['a']]],
            [[['aa']]],
            [[['aa']], [['aaab']], [['aa']]]
        ]
        mock_to_vector.side_effect = expected
        extractor = ReadabilityMeasures()
        assert_almost_equal(extractor.transform(self.documents),
                            np.array(expected))
        calls = [call(doc) for doc in self.documents]
        mock_tokenize_content.assert_has_calls(calls)
        calls = [call(tok) for tok in mock_tokenize_content.side_effect]
        mock_to_vector.assert_has_calls(calls)

    def test_empty_content(self, mock_tokenize_content, mock_to_vector):
        expected = [np.array([ReadabilityMeasures.INF])]
        mock_tokenize_content.side_effect = [[]]
        mock_to_vector.side_effect = expected
        extractor = ReadabilityMeasures()
        assert_almost_equal(extractor.transform(['..']), np.array(expected))


class TestTokenizeContent(TestCase):
    def setUp(self):
        self.content = 'Ani Budi Cika. Ani Cika.\nBudi.\n'
        p1 = patch('otdet.feature_extraction.sent_tokenize')
        p2 = patch('otdet.feature_extraction.word_tokenize')
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        self.mock_sent_tokenize = p1.start()
        self.mock_word_tokenize = p2.start()
        self.mock_sent_tokenize.return_value = ['Ani Budi Cika.', 'Ani Cika.',
                                                'Budi']
        self.mock_word_tokenize.side_effect = [
            ['Ani', 'Budi', 'Cika', '.'],
            ['Ani', 'Cika', '.'],
            ['Budi', '.']
        ]

    def test_default(self):
        extractor = ReadabilityMeasures()
        expected = [
            ['Ani', 'Budi', 'Cika'],
            ['Ani', 'Cika'],
            ['Budi']
        ]
        assert_equal(extractor._tokenize_content(self.content), expected)
        self.mock_sent_tokenize.assert_called_with(self.content)
        calls = [call(sent) for sent in self.mock_sent_tokenize.return_value]
        self.mock_word_tokenize.assert_has_calls(calls)

    def test_no_remove_punct(self):
        extractor = ReadabilityMeasures(remove_punct=False)
        expected = [
            ['Ani', 'Budi', 'Cika', '.'],
            ['Ani', 'Cika', '.'],
            ['Budi', '.']
        ]
        assert_equal(extractor._tokenize_content(self.content), expected)

    def test_all_punct(self):
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
        tokenized_content = [['1st', 'sent'], ['2nd', 'sent']]
        extractor = ReadabilityMeasures(measures=['smog', 'ari', 'lix'])
        expected = np.arange(1, 4)
        result = extractor._to_vector(tokenized_content)
        assert_almost_equal(result, expected)
        mock_lix.assert_called_with(tokenized_content)
        mock_ari.assert_called_with(tokenized_content)
        mock_smog.assert_called_with(tokenized_content)


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
        calls = [call(w) for s in tokenized_content for w in s]
        mock_num_syllables.assert_has_calls(calls)


class TestTotalChars:
    def test_default(self):
        tokenized_content = [
            ['ani', 'budi', 'cika'],
            ['ani', 'cika'],
            ['budi']
        ]
        extractor = ReadabilityMeasures()
        assert_equal(extractor.total_chars(tokenized_content), 22)


@patch.object(ReadabilityMeasures, 'total_sylls', return_value=50)
class TestFleschease:
    def setUp(self):
        self.tokenized_content = Mock(spec=TokenizedContent)
        self.tokenized_content.num_words = 30
        self.tokenized_content.num_sents = 5

    def test_default(self, mock_total_sylls):
        result = ReadabilityMeasures.fleschease(self.tokenized_content)
        assert_almost_equal(result, 59.745)
        mock_total_sylls.assert_called_with(self.tokenized_content)

    def test_zero_words(self, mock_total_sylls):
        self.tokenized_content.num_words = 0
        result = ReadabilityMeasures.fleschease(self.tokenized_content)
        assert_almost_equal(result, ReadabilityMeasures.INF)

    def test_zero_sents(self, mock_total_sylls):
        self.tokenized_content.num_sents = 0
        result = ReadabilityMeasures.fleschease(self.tokenized_content)
        assert_almost_equal(result, ReadabilityMeasures.INF)


@patch.object(ReadabilityMeasures, 'total_sylls', return_value=50)
class TestFleschgrade:
    def setUp(self):
        self.tokenized_content = Mock(spec=TokenizedContent)
        self.tokenized_content.num_words = 30
        self.tokenized_content.num_sents = 5

    def test_default(self, mock_total_sylls):
        result = ReadabilityMeasures.fleschgrade(self.tokenized_content)
        assert_almost_equal(result, 6.41666667)
        mock_total_sylls.assert_called_with(self.tokenized_content)

    def test_zero_words(self, mock_total_sylls):
        self.tokenized_content.num_words = 0
        result = ReadabilityMeasures.fleschgrade(self.tokenized_content)
        assert_almost_equal(result, ReadabilityMeasures.INF)

    def test_zero_sents(self, mock_total_sylls):
        self.tokenized_content.num_sents = 0
        result = ReadabilityMeasures.fleschgrade(self.tokenized_content)
        assert_almost_equal(result, ReadabilityMeasures.INF)


@patch.object(ReadabilityMeasures, 'num_syllables')
class TestFogindex:
    def setUp(self):
        self.tokenized_content = MagicMock(spec=TokenizedContent)
        self.tokenized_content.__iter__.return_value = iter([list('ab'),
                                                             list('cdefg')])
        self.tokenized_content.num_words = 30
        self.tokenized_content.num_sents = 5

    def test_default(self, mock_num_syllables):
        mock_num_syllables.side_effect = [8, 2, 3, 1, 5, 6, 7]
        result = ReadabilityMeasures.fogindex(self.tokenized_content)
        assert_almost_equal(result, 22.66666667)
        calls = [call(w) for s in self.tokenized_content for w in s]
        mock_num_syllables.assert_has_calls(calls)

    def test_zero_words(self, mock_num_syllables):
        mock_num_syllables.side_effect = [8, 2, 3, 1, 5, 6, 7]
        self.tokenized_content.num_words = 0
        result = ReadabilityMeasures.fogindex(self.tokenized_content)
        assert_almost_equal(result, ReadabilityMeasures.INF)

    def test_zero_sents(self, mock_num_syllables):
        mock_num_syllables.side_effect = [8, 2, 3, 1, 5, 6, 7]
        self.tokenized_content.num_sents = 0
        result = ReadabilityMeasures.fogindex(self.tokenized_content)
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestColemanliau:
    def setUp(self):
        self.tokenized_content = Mock(spec=TokenizedContent)
        self.tokenized_content.num_chars = 100
        self.tokenized_content.num_words = 30
        self.tokenized_content.num_sents = 5

    def test_default(self):
        result = ReadabilityMeasures.colemanliau(self.tokenized_content)
        assert_almost_equal(result, 3.83283333)

    def test_zero_words(self):
        self.tokenized_content.num_words = 0
        result = ReadabilityMeasures.colemanliau(self.tokenized_content)
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestAri:
    def setUp(self):
        self.tokenized_content = Mock(spec=TokenizedContent)
        self.tokenized_content.num_chars = 100
        self.tokenized_content.num_words = 30
        self.tokenized_content.num_sents = 5

    def test_default(self):
        result = ReadabilityMeasures.ari(self.tokenized_content)
        assert_almost_equal(result, -2.72999999)

    def test_zero_words(self):
        self.tokenized_content.num_words = 0
        result = ReadabilityMeasures.ari(self.tokenized_content)
        assert_almost_equal(result, ReadabilityMeasures.INF)

    def test_zero_sents(self):
        self.tokenized_content.num_sents = 0
        result = ReadabilityMeasures.ari(self.tokenized_content)
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestLix:
    def setUp(self):
        self.tokenized_content = [
            ['a', 'aaaaaa'],
            ['aa', 'aaa', 'aaaaaaaa', 'aa', 'aaaaaaa']
        ]

    @patch.object(ReadabilityMeasures, 'total_sents')
    @patch.object(ReadabilityMeasures, 'total_words')
    def test_default(self, mock_total_words, mock_total_sents):
        mock_total_words.return_value = 30
        mock_total_sents.return_value = 5
        result = ReadabilityMeasures.lix(self.tokenized_content)
        assert_almost_equal(result, 16)
        mock_total_words.assert_called_with(self.tokenized_content)
        mock_total_sents.assert_called_with(self.tokenized_content)

    @patch.object(ReadabilityMeasures, 'total_words')
    def test_zero_words(self, mock_total_words):
        mock_total_words.return_value = 0
        result = ReadabilityMeasures.lix(self.tokenized_content)
        assert_almost_equal(result, ReadabilityMeasures.INF)

    @patch.object(ReadabilityMeasures, 'total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.lix(self.tokenized_content)
        assert_almost_equal(result, ReadabilityMeasures.INF)


class TestSmog:
    def setUp(self):
        self.tokenized_content = [['a', 'b'], ['c', 'd', 'e', 'f', 'g']]

    @patch.object(ReadabilityMeasures, 'total_sents')
    @patch.object(ReadabilityMeasures, 'num_syllables')
    def test_default(self, mock_num_syllables, mock_total_sents):
        mock_num_syllables.side_effect = [8, 2, 3, 1, 5, 6, 7]
        mock_total_sents.return_value = 5
        result = ReadabilityMeasures.smog(self.tokenized_content)
        assert_almost_equal(result, 8.47722557)
        calls = [call(w) for s in self.tokenized_content for w in s]
        mock_num_syllables.assert_has_calls(calls)
        mock_total_sents.assert_called_with(self.tokenized_content)

    @patch.object(ReadabilityMeasures, 'total_sents')
    def test_zero_sents(self, mock_total_sents):
        mock_total_sents.return_value = 0
        result = ReadabilityMeasures.smog(self.tokenized_content)
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
        mock_avg_syllables.assert_called_with(1)


@patch.object(ReadabilityMeasures, 'd', d)
class TestAvgSyllables:
    def test_word_len_exist_in_corpus(self):
        assert_almost_equal(ReadabilityMeasures.avg_syllables(1), 7/6)

    def test_word_len_not_exist_in_corpus(self):
        assert_almost_equal(ReadabilityMeasures.avg_syllables(3), 5/4)
