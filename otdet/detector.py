"""
Out-of-topic post detection methods.
"""

import numpy as np
import scipy.spatial.distance as dist
from sklearn.cross_validation import LeaveOneOut

from otdet.feature_extraction import CountVectorizerWrapper


class OOTDetector:
    """Off-topic detection methods."""

    def __init__(self, extractor=None):
        if extractor is None:
            self.extractor = CountVectorizerWrapper(input='content',
                                                    stop_words='english')
        else:
            self.extractor = extractor

    def design_matrix(self, documents):
        """Returns feature vector of each document as matrix."""
        return self.extractor.fit_transform(documents)

    def clust_dist(self, documents, metric='euclidean'):
        """Compute ClustDist score of each document."""
        X = self.design_matrix(documents)
        return np.mean(dist.squareform(dist.pdist(X, metric)), axis=0)

    def mean_comp(self, documents, metric='euclidean'):
        """Compute MeanComp score of each document."""
        X = self.design_matrix(documents)
        m = X.shape[0]
        res = np.zeros(m)
        for i, (comp, vec) in enumerate(LeaveOneOut(m)):
            v, u = np.mean(X[comp], axis=0), np.ravel(X[vec])
            distfunc = getattr(dist, metric)
            res[i] = distfunc(u, v)
        return res

    def txt_comp_dist(self, documents, metric='euclidean'):
        """Compute TxtCompDist score of each document."""
        self.extractor.fit(documents)
        res = []
        for i, cont in enumerate(documents):
            comp = ' '.join(documents[:i] + documents[i+1:])
            u = self.extractor.transform([cont])
            v = self.extractor.transform([comp])
            distfunc = getattr(dist, metric)
            res.append(distfunc(u, v))
        return np.array(res)
