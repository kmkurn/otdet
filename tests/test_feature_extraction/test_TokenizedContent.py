from nose.tools import assert_equal

from unittest.mock import call, patch

from otdet.feature_extraction import TokenizedContent


@patch('otdet.feature_extraction.sent_tokenize')
@patch('otdet.feature_extraction.word_tokenize')
class TestInit:
    def test_default(self, mock_word_tokenize, mock_sent_tokenize):
        mock_sent_tokenize.return_value = ['a b c.', 'a b.', 'b b c']
        mock_word_tokenize.side_effect = [['a', 'b', 'c', '.'],
                                          ['a', 'b', '.'], ['b', 'b', 'c']]
        content = 'a b c.\na b. b b c.\n\n'
        expected = [['a', 'b', 'c'], ['a', 'b'], ['b', 'b', 'c']]
        tc = TokenizedContent(content)
        assert_equal(tc._tokcont, expected)
        mock_sent_tokenize.assert_called_with(content)
        calls = [call(s) for s in mock_sent_tokenize.return_value]
        mock_word_tokenize.assert_has_calls(calls)

    def test_no_remove_punc(self, mock_word_tokenize, mock_sent_tokenize):
        expected = [['a', 'b', 'c', '.'], ['a', 'b', '.'], ['b', 'b', 'c']]
        mock_sent_tokenize.return_value = ['a b c.', 'a b.', 'b b c']
        mock_word_tokenize.side_effect = expected
        content = 'a b c.\na b. b b c.\n\n'
        tc = TokenizedContent(content, remove_punct=False)
        assert_equal(tc._tokcont, expected)


content = 'a b c.\na, b.'
tokcont = [['a', 'b', 'c'], ['a', 'b']]


class TestNumWords:
    def test_default(self):
        tc = TokenizedContent(content)
        tc._tokcont = tokcont
        assert_equal(tc.num_words, 5)

    def test_no_words(self):
        tc = TokenizedContent(content)
        tc._tokcont = []
        assert_equal(tc.num_words, 0)
