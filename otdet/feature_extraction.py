from functools import lru_cache
from statistics import mean
from string import punctuation
import warnings

from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from otdet.util import lazyproperty


class ReadabilityMeasures:
    """Extract features based on readablility measures."""

    d = cmudict.dict()
    INF = 10**9

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

        tokcontents = [TokenizedContent(cont, self.remove_punct)
                       for cont in contents]
        return np.array([self._to_vector(tcont) for tcont in tokcontents])

    def _to_vector(self, tokenized_content):
        """Convert a tokenized content to a feature vector."""
        return np.array([getattr(self, m)(tokenized_content)
                        for m in self.measures])

    @classmethod
    def fleschease(cls, tokenized_content):
        """Return the Flesch-Kindaid Reading Ease measure."""
        nwords = tokenized_content.num_words
        nsents = tokenized_content.num_sents
        nsylls = cls.total_sylls(tokenized_content)
        try:
            return 206.835 - 1.015*(nwords/nsents) - 84.6*(nsylls/nwords)
        except ZeroDivisionError:
            return cls.INF

    @classmethod
    def fleschgrade(cls, tokenized_content):
        """Return the Flesch-Kinaid Grade Level measure."""
        nwords = tokenized_content.num_words
        nsents = tokenized_content.num_sents
        nsylls = cls.total_sylls(tokenized_content)
        try:
            return 11.8*(nsylls/nwords) + 0.39*(nwords/nsents) - 15.59
        except ZeroDivisionError:
            return cls.INF

    @classmethod
    def fogindex(cls, tokenized_content):
        """Return the Gunning-Fog index."""
        nwords = tokenized_content.num_words
        nsents = tokenized_content.num_sents
        nwords3sylls = sum(sum(cls.num_syllables(w) >= 3 for w in s)
                           for s in tokenized_content)
        try:
            return (nwords/nsents) + (nwords3sylls/nwords)*100
        except ZeroDivisionError:
            return cls.INF

    @classmethod
    def colemanliau(cls, tokenized_content):
        """Return the Coleman-Liau formula."""
        nchars = tokenized_content.num_chars
        nwords = tokenized_content.num_words
        nsents = tokenized_content.num_sents
        try:
            return 5.89*(nchars/nwords) - 0.3*(nsents/(nwords*100)) - 15.8
        except ZeroDivisionError:
            return cls.INF

    @classmethod
    def ari(cls, tokenized_content):
        """Return the Automated Readability Index."""
        nchars = tokenized_content.num_chars
        nwords = tokenized_content.num_words
        nsents = tokenized_content.num_sents
        try:
            return 4.71*(nchars/nwords) + 0.5*(nwords/nsents) - 21.43
        except ZeroDivisionError:
            return cls.INF

    @classmethod
    def lix(cls, tokenized_content):
        """Return the Lix formula."""
        nwords = tokenized_content.num_words
        nwords6chars = sum(sum(len(w) >= 6 for w in s)
                           for s in tokenized_content)
        nsents = tokenized_content.num_sents
        try:
            return (nwords/nsents) + 100*(nwords6chars/nwords)
        except ZeroDivisionError:
            return cls.INF

    @classmethod
    def smog(cls, tokenized_content):
        """Return the SMOG index."""
        nwords3sylls = sum(sum(cls.num_syllables(w) >= 3 for w in s)
                           for s in tokenized_content)
        nsents = tokenized_content.num_sents
        try:
            return 3 + ((nwords3sylls*30)/nsents)**0.5
        except ZeroDivisionError:
            return cls.INF

    @staticmethod
    def total_sylls(tokenized_content):
        """Return the total number of syllables in a tokenized content."""
        return sum(sum(ReadabilityMeasures.num_syllables(w) for w in s)
                   for s in tokenized_content)

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


class TokenizedContent:
    """Class representing a tokenized content."""

    def __init__(self, content, remove_punct=True):
        self._tokcont = [word_tokenize(s) for s in sent_tokenize(content)]
        if remove_punct:
            self._tokcont = [[w for w in s if w not in punctuation]
                             for s in self._tokcont[:]]
        # Remove zero-length sentence
        self._tokcont = [s for s in self._tokcont[:] if len(s) > 0]


    @lazyproperty
    def num_sents(self):
        """Return the total number of sentences."""
        return len(self._tokcont)

    @lazyproperty
    def num_words(self):
        """Return the total number of words."""
        return sum(len(s) for s in self._tokcont)

    @lazyproperty
    def num_chars(self):
        """Return the total number of chars."""
        return sum(sum(len(w) for w in s) for s in self._tokcont)


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
