from unittest.mock import call, patch, Mock, MagicMock

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
        self.tokenized_content = MagicMock(spec=TokenizedContent)
        self.tokenized_content.__iter__.return_value = iter([
            ['a', 'aaaaaa'],
            ['aa', 'aaa', 'aaaaaaaa', 'aa', 'aaaaaaa']
        ])
        self.tokenized_content.num_words = 30
        self.tokenized_content.num_sents = 5

    def test_default(self):
        result = ReadabilityMeasures.lix(self.tokenized_content)
        assert_almost_equal(result, 16)

    def test_zero_words(self):
        self.tokenized_content.num_words = 0
        result = ReadabilityMeasures.lix(self.tokenized_content)
        assert_almost_equal(result, ReadabilityMeasures.INF)

    def test_zero_sents(self):
        self.tokenized_content.num_sents = 0
        result = ReadabilityMeasures.lix(self.tokenized_content)
        assert_almost_equal(result, ReadabilityMeasures.INF)


@patch.object(ReadabilityMeasures, 'num_syllables')
class TestSmog:
    def setUp(self):
        self.tokenized_content = MagicMock(spec=TokenizedContent)
        self.tokenized_content.__iter__.return_value = iter([
            ['a', 'b'], ['c', 'd', 'e', 'f', 'g']
        ])
        self.tokenized_content.num_sents = 5

    def test_default(self, mock_num_syllables):
        mock_num_syllables.side_effect = [8, 2, 3, 1, 5, 6, 7]
        result = ReadabilityMeasures.smog(self.tokenized_content)
        assert_almost_equal(result, 8.47722557)
        calls = [call(w) for s in self.tokenized_content for w in s]
        mock_num_syllables.assert_has_calls(calls)

    def test_zero_sents(self, mock_num_syllables):
        mock_num_syllables.side_effect = [8, 2, 3, 1, 5, 6, 7]
        self.tokenized_content.num_sents = 0
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
