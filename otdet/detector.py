"""
Anomalous text detection methods.
"""

import numpy as np
import scipy.spatial.distance as dist
from sklearn.cross_validation import LeaveOneOut
from sklearn.feature_extraction.text import CountVectorizer

from otdet.util import lazyproperty


class OffTopicDetector:
    """Off-topic detection methods."""

    def __init__(self, filenames):
        self.filenames = filenames

    @lazyproperty
    def design_matrix(self):
        """Returns feature vector of each document as matrix."""
        vectorizer = CountVectorizer(input='filename', stop_words='english')
        return vectorizer.fit_transform(self.filenames).toarray()

    @lazyproperty
    def contents(self):
        """Returns the content of each document."""
        res = []
        for filename in self.filenames:
            f = open(filename)
            res.append(f.read())
            f.close()
        return res

    def clust_dist(self, metric='euclidean'):
        """Compute ClustDist score of each document."""
        X = self.design_matrix
        return np.mean(dist.squareform(dist.pdist(X, metric)), axis=0)

    def mean_comp(self, metric='euclidean'):
        """Compute MeanComp score of each document."""
        X = self.design_matrix
        m = X.shape[0]
        res = np.zeros(m)
        for i, (comp, vec) in enumerate(LeaveOneOut(m)):
            v, u = np.mean(X[comp], axis=0), np.ravel(X[vec])
            distfunc = getattr(dist, metric)
            res[i] = distfunc(u, v)
        return res

    def txt_comp_dist(self, metric='euclidean'):
        """Compute TxtCompDist score of each document."""
        vectorizer = CountVectorizer(stop_words='english')
        vectorizer.fit(self.contents)
        res = []
        for i, cont in enumerate(self.contents):
            comp = ' '.join(self.contents[:i] + self.contents[i+1:])
            u = vectorizer.transform([cont]).toarray()
            v = vectorizer.transform([comp]).toarray()
            distfunc = getattr(dist, metric)
            res.append(distfunc(u, v))
        return np.array(res)
