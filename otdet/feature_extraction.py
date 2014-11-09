from functools import lru_cache
from statistics import mean
from string import punctuation
import warnings

from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


class ReadabilityMeasures:
    """Extract features based on readablility measures."""

    d = cmudict.dict()

    def __init__(self, lowercase=True, remove_punct=True, measures=None,
                 **kwargs):
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        if measures is None:
            self.measures = [
                'fleschease', 'fleschgrade', 'fogindex', 'colemanliau',
                'ari', 'lix', 'smog'
            ]
        else:
            self.measures = measures

    def fit(self, *args, **kwargs):
        # Do nothing
        pass

    def fit_transform(self, documents):
        # Directly transform
        return self.transform(documents)

    def transform(self, documents):
        """Transform documents into vectors of readability measures."""

        if self.lowercase:
            contents = [doc.lower() for doc in documents]
        else:
            contents = documents

        tokcontents = [self._tokenize_content(cont) for cont in contents]
        return np.array([self._to_vector(tcont) for tcont in tokcontents])

    def _tokenize_content(self, content):
        """Tokenize content into list of word lists."""
        res = [word_tokenize(s) for s in sent_tokenize(content)]
        if self.remove_punct:
            res = [[w for w in s if w not in punctuation] for s in res[:]]
        # Remove empty words/sentence
        res = [s for s in res[:] if len(s) > 0]
        res = [[w for w in s if len(w) > 0] for s in res[:]]
        return res

    def _to_vector(self, tokenized_content):
        """Convert a tokenized content to a feature vector."""
        return np.array([getattr(self, m)(tokenized_content)
                        for m in self.measures])

    @classmethod
    def fleschease(cls, tokenized_content):
        """Return the Flesch-Kindaid Reading Ease measure."""
        nwords = cls.total_words(tokenized_content)
        nsents = cls.total_sents(tokenized_content)
        nsylls = cls.total_sylls(tokenized_content)
        return 206.835 - 1.015*(nwords/nsents) - 84.6*(nsylls/nwords)

    @classmethod
    def fleschgrade(cls, tokenized_content):
        """Return the Flesch-Kinaid Grade Level measure."""
        nwords = cls.total_words(tokenized_content)
        nsents = cls.total_sents(tokenized_content)
        nsylls = cls.total_sylls(tokenized_content)
        return 11.8*(nsylls/nwords) + 0.39*(nwords/nsents) - 15.59

    @classmethod
    def fogindex(cls, tokenized_content):
        """Return the Gunning-Fog index."""
        nwords = cls.total_words(tokenized_content)
        nsents = cls.total_sents(tokenized_content)
        nwords3sylls = sum(sum(cls.num_syllables(w) >= 3 for w in s)
                           for s in tokenized_content)
        return (nwords/nsents) + (nwords3sylls/nwords)*100

    @classmethod
    def colemanliau(cls, tokenized_content):
        """Return the Coleman-Liau formula."""
        nchars = cls.total_chars(tokenized_content)
        nwords = cls.total_words(tokenized_content)
        nsents = cls.total_sents(tokenized_content)
        return 5.89*(nchars/nwords) - 0.3*(nsents/(nwords*100)) - 15.8

    @classmethod
    def ari(cls, tokenized_content):
        """Return the Automated Readability Index."""
        nchars = cls.total_chars(tokenized_content)
        nwords = cls.total_words(tokenized_content)
        nsents = cls.total_sents(tokenized_content)
        return 4.71*(nchars/nwords) + 0.5*(nwords/nsents) - 21.43

    @classmethod
    def lix(cls, tokenized_content):
        """Return the Lix formula."""
        nwords = cls.total_words(tokenized_content)
        nwords6chars = sum(sum(len(w) >= 6 for w in s)
                           for s in tokenized_content)
        nsents = cls.total_sents(tokenized_content)
        return (nwords/nsents) + 100*(nwords6chars/nwords)

    @classmethod
    def smog(cls, tokenized_content):
        """Return the SMOG index."""
        nwords3sylls = sum(sum(cls.num_syllables(w) >= 3 for w in s)
                           for s in tokenized_content)
        nsents = cls.total_sents(tokenized_content)
        return 3 + ((nwords3sylls*30)/nsents)**0.5

    @staticmethod
    def total_sents(tokenized_content):
        """Return the total number of sentences in a tokenized content."""
        return len(tokenized_content)

    @staticmethod
    def total_words(tokenized_content):
        """Return the total number of words in a tokenized content."""
        return sum(len(s) for s in tokenized_content)

    @staticmethod
    def total_sylls(tokenized_content):
        """Return the total number of syllables in a tokenized content."""
        return sum(sum(ReadabilityMeasures.num_syllables(w) for w in s)
                   for s in tokenized_content)

    @staticmethod
    def total_chars(tokenized_content):
        """Return the total number of characters in a tokenized content."""
        return sum(sum(len(w) for w in s) for s in tokenized_content)

    @staticmethod
    @lru_cache(maxsize=None)
    def num_syllables(word):
        """Return the number of syllables in a word."""
        if word in ReadabilityMeasures.d:
            res = mean(len([y for y in x if y[-1].isdigit()])
                       for x in ReadabilityMeasures.d[word])
        else:
            warnings.warn("No '{}' found in CMU corpus".format(word))
            res = ReadabilityMeasures.avg_syllables(len(word))
        return res

    @staticmethod
    @lru_cache(maxsize=None)
    def avg_syllables(wordlen):
        """Return the avg number of syllables of words with given length."""
        res = [ReadabilityMeasures.num_syllables(w)
               for w in ReadabilityMeasures.d if len(w) == wordlen]
        if len(res) == 0:
            res = [ReadabilityMeasures.num_syllables(w)
                   for w in ReadabilityMeasures.d]
        return mean(res)


class CountVectorizerWrapper(CountVectorizer):
    """Wrapper around CountVectorizer class in scikit-learn."""

    def __init__(self, *args, **kwargs):
        super(CountVectorizerWrapper, self).__init__(*args, **kwargs)

    def fit_transform(self, *args, **kwargs):
        """Wrapper around fit_transform() method in CountVectorizer."""
        r = super(CountVectorizerWrapper, self).fit_transform(*args, **kwargs)
        return r.toarray()

    def transform(self, *args, **kwargs):
        """Wrapper around transform() method in CountVectorizer."""
        r = super(CountVectorizerWrapper, self).transform(*args, **kwargs)
        return r.toarray()
