#!/usr/bin/env python

import argparse
from glob import glob
import itertools as it
from multiprocessing import Pool
import os
import os.path
import sys

import numpy as np
import pandas as pd

from otdet.detector import OOTDetector
from otdet.evaluation import TopListEvaluator
from otdet.util import pick


def experiment(setting):
    """Do experiment with the specified setting."""
    norm_dir, oot_dir, num_norm, num_oot, method, metric, num_top = setting
    result = []
    for jj in range(args.niter):
        # Obtain normal posts
        norm_files = pick(glob(os.path.join(norm_dir, '*.txt')), k=num_norm,
                          randomized=False)
        # Obtain OOT posts
        oot_files = pick(glob(os.path.join(oot_dir, '*.txt')), k=num_oot)
        # Combine them both
        files = norm_files + oot_files
        is_oot = [False]*len(norm_files) + [True]*len(oot_files)

        # Apply OOT post detection methods
        detector = OOTDetector(files)
        methodfunc = getattr(detector, method)
        distances = methodfunc(metric=metric)

        # Construct ranked list of OOT posts (1: most off-topic)
        # In case of tie, prioritize normal post (worst case)
        s = sorted(zip(distances, is_oot), key=lambda x: x[1])
        subresult = sorted(s, reverse=True)

        # Append to result
        result.append(subresult)
    return result


def evaluate(result, setting):
    """Evaluate an experiment result with the given setting."""
    *_, num_top = setting
    evaluator = TopListEvaluator(result, N=num_top)
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
    settings = list(it.product(args.norm_dir, args.oot_dir, args.num_norm,
                               args.num_oot, args.method, args.metric,
                               args.num_top))

    # Do experiments
    with Pool(processes=args.jobs) as pool:
        results = pool.map_async(experiment, settings).get()

    index_tup, column_tup = [], []
    data = np.array([])
    for setting, result in zip(settings, results):
        # Evaluate the result of each setting
        baseline, performance, min_sup, max_sup = evaluate(result, setting)

        # Prepare Pandas MultiIndex tuples
        (norm_dir, oot_dir, num_norm, num_oot,
            method, metric, num_top) = setting
        norm_dir = shorten(norm_dir)
        oot_dir = shorten(oot_dir)
        index_tup.append((norm_dir, oot_dir, method, metric))
        for res in ['base', 'perf']:
            for k in range(min_sup, max_sup+1):
                column_tup.append((num_norm, num_oot, num_top, res, k))

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
    index_names = ['norm_dir', 'oot_dir', 'method', 'metric']
    column_names = ['num_norm', 'num_oot', 'num_top', 'result', 'k']
    index = pd.MultiIndex.from_tuples(index, names=index_names)
    columns = pd.MultiIndex.from_tuples(columns, names=column_names)
    df = pd.DataFrame(data.reshape((len(index), len(columns))),
                      index=index, columns=columns)
    # Store in HDF5 format
    df.to_hdf(args.hdf_name, args.hdf_key)
    print("Stored in HDF5 format with the name '{}'".format(args.hdf_key))

    print('Done', file=sys.stderr, flush=True)
