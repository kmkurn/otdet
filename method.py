"""
Anomalous text detection methods.
"""

import numpy as np
import scipy.spatial.distance as dist
from sklearn.cross_validation import LeaveOneOut


def clust_dist(X, metric='euclidean'):
    """Compute the average distance of a vector to others.

    ClustDist of a vector x to a set of vectors V is defined as the average
    distance of x to all vectors v in V.

    Parameters
    ----------
    X : matrix
        An m-by-n matrix which represents m n-dimensional vectors.

    metric : string, optional (default = 'euclidean')
        Distance metric to be used. See scipy.spatial.distance for a list
        of available values.

    Returns
    -------
    dists : array
        An array of average distances where dists[i] represents the average
        distance between vector X[i] and other vectors in X.

    Examples
    --------
    >>> import numpy as np
    >>> from method import clust_dist
    >>> X = np.array([[1, 2], [-1, 0], [5, 3]])
    >>> clust_dist(X, metric='euclidean')
    array([ 2.31717758,  3.17887702,  3.61043652])

    References
    ----------
    .. [1] Guthrie, D. (2008).
       Unsupervised Detection of Anomalous Text.
       University of Sheffield.
       Retrieved from http://nlp.shef.ac.uk/talks/Guthrie_20081127.pdf
    """
    return np.mean(dist.squareform(dist.pdist(X, metric)), axis=0)


def mean_comp(X, metric='euclidean'):
    """Compute the distance between each vector to its complement vector."""
    m = X.shape[0]
    res = np.zeros(m)
    for i, (comp, vec) in enumerate(LeaveOneOut(m)):
        v, u = np.mean(X[comp], axis=0), np.ravel(X[vec])
        distfunc = getattr(dist, metric)
        res[i] = distfunc(u, v)
    return res
