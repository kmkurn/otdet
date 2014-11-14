#!/usr/bin/env python

import argparse
from collections import namedtuple
from glob import glob
import itertools as it
import os
import os.path
import sys

import numpy as np
import pandas as pd

from otdet.detector import OOTDetector
from otdet.evaluation import TopListEvaluator
from otdet.feature_extraction import ReadabilityMeasures
from otdet.util import pick


def experiment(setting, niter):
    """Do experiment with the specified setting."""
    # Obtain normal posts
    norm_files = pick(glob(os.path.join(setting.norm_dir, '*.txt')),
                      k=setting.num_norm, randomized=False)
    norm_docs = []
    for file in norm_files:
        with open(file) as f:
            norm_docs.append(f.read())

    res = []
    for jj in range(niter):
        # Obtain OOT posts
        oot_files = pick(glob(os.path.join(setting.oot_dir, '*.txt')),
                         k=setting.num_oot)
        oot_docs = []
        for file in oot_files:
            with open(file) as f:
                oot_docs.append(f.read())
        # Combine them both
        documents = norm_docs + oot_docs
        is_oot = [False]*setting.num_norm + [True]*setting.num_oot

        # Apply OOT post detection methods
        if setting.feature == 'unigram':
            detector = OOTDetector()
        else:
            extractor = ReadabilityMeasures()
            detector = OOTDetector(extractor=extractor)
        func = getattr(detector, setting.method)
        distances = func(documents, metric=setting.metric)

        # Construct ranked list of OOT posts (1: most off-topic)
        # In case of tie, prioritize normal post (worst case)
        s = sorted(zip(distances, is_oot), key=lambda x: x[1])
        subresult = sorted(s, reverse=True)
        res.append(subresult)
    return res


def evaluate(result, setting):
    """Evaluate an experiment result with the given setting."""
    evaluator = TopListEvaluator(result, N=setting.num_top)
    return (evaluator.baseline, evaluator.performance,
            evaluator.min_sup, evaluator.max_sup)


def shorten(dirname):
    """Shorten a thread directory name."""
    split_path = dirname.split(os.sep)
    thread, post = (split_path[-2], split_path[-1]) if split_path[-1] != '' \
        else (split_path[-3], split_path[-2])
    post_id = post.split('__')[0]
    return thread[:3] + post_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run experiment with given '
                                     'settings')
    parser.add_argument('-nd', '--norm-dir', type=str, nargs='+',
                        required=True, help='Normal thread directory')
    parser.add_argument('-od', '--oot-dir', type=str, nargs='+', required=True,
                        help='Thread directory from which '
                        'OOT post will be taken')
    parser.add_argument('-m', '--num-norm', type=int, nargs='+', required=True,
                        help='Number of posts taken from '
                        'normal thread directory')
    parser.add_argument('-n', '--num-oot', type=int, nargs='+', required=True,
                        help='Number of posts taken from '
                        'another thread directory to be OOT posts')
    parser.add_argument('-a', '--method', type=str, nargs='+', required=True,
                        choices=['clust_dist', 'mean_comp', 'txt_comp_dist'],
                        help='OOT post detection method to use')
    parser.add_argument('-d', '--metric', type=str, nargs='+', required=True,
                        choices=['euclidean', 'cityblock', 'cosine',
                                 'correlation'],
                        help='Distance metric to use')
    parser.add_argument('-f', '--feature', type=str, nargs='+', required=True,
                        choices=['unigram', 'readability'],
                        help='Text features to be used')
    parser.add_argument('-t', '--num-top', type=int, nargs='+', required=True,
                        help='Number of posts in top N list')
    parser.add_argument('--niter', type=int, default=1,
                        help='Number of iteration for each method')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of work processes')
    parser.add_argument('--hdf-name', type=str, required=True,
                        help='Where to store the result in HDF5 format')
    parser.add_argument('--hdf-key', type=str, default='df',
                        help='Identifier in the HDF5 store')
    args = parser.parse_args()

    # Experiment settings
    names = ['method', 'feature', 'metric', 'norm_dir', 'oot_dir', 'num_norm',
             'num_oot', 'num_top']
    ExprSetting = namedtuple('ExprSetting', names)
    settings = list(it.product(args.method, args.feature, args.metric,
                               args.norm_dir, args.oot_dir, args.num_norm,
                               args.num_oot, args.num_top))
    settings = [ExprSetting(*sett) for sett in settings[:]]

    # Do experiments
    results = (experiment(setting, args.niter) for setting in settings)

    index_tup, column_tup = [], []
    data = np.array([])
    for setting, result in zip(settings, results):
        # Evaluate the result of each setting
        baseline, performance, min_sup, max_sup = evaluate(result, setting)

        # Prepare Pandas MultiIndex tuples
        norm_dir = shorten(setting.norm_dir)
        oot_dir = shorten(setting.oot_dir)
        index_tup.append((setting.method, setting.feature, setting.metric,
                          norm_dir, oot_dir))
        for res in ['base', 'perf']:
            for k in range(min_sup, max_sup+1):
                column_tup.append((setting.num_norm, setting.num_oot,
                                   setting.num_top, res, k))

        # Prepare Pandas DataFrame data
        data = np.concatenate((data, baseline))
        data = np.concatenate((data, performance))

    # Create index tuples list
    st = set()
    index = []
    for idx in index_tup:
        if idx not in st:
            index.append(idx)
            st.add(idx)
    # Create column tuples list
    st = set()
    columns = []
    for col in column_tup:
        if col not in st:
            columns.append(col)
            st.add(col)

    # Prepare to store in HDF5 format
    index_names = names[:5]
    column_names = names[5:] + ['result', 'k']
    index = pd.MultiIndex.from_tuples(index, names=index_names)
    columns = pd.MultiIndex.from_tuples(columns, names=column_names)
    df = pd.DataFrame(data.reshape((len(index), len(columns))),
                      index=index, columns=columns)

    # Store in HDF5 format
    df.to_hdf(args.hdf_name, args.hdf_key)
    print("Stored in HDF5 format with the name '{}'".format(args.hdf_key))

    print('Done', file=sys.stderr, flush=True)
