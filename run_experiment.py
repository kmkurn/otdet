#!/usr/bin/env python

import argparse
from collections import defaultdict
from glob import glob
import itertools as it
from multiprocessing import Pool
import os.path
import sys

import numpy as np

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
    num_norm, num_oot, num_top = setting[2], setting[3], setting[6]
    evaluator = TopListEvaluator(M=num_norm+num_oot, n=num_oot, N=num_top)
    return (evaluator.baseline, evaluator.baseline_skew,
            evaluator.get_performance(result),
            evaluator.get_performance_skew(result))


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
    args = parser.parse_args()

    # Experiment settings
    settings = list(it.product(args.norm_dir, args.oot_dir, args.num_norm,
                               args.num_oot, args.method, args.metric,
                               args.num_top))

    # Do experiments
    with Pool(processes=args.jobs) as pool:
        results = pool.map_async(experiment, settings).get()

    # Store the report
    report = defaultdict(dict)
    for setting, result in zip(settings, results):
        base, base_skew, perf, perf_skew = evaluate(result, setting)
        report[setting]['baseline'] = base
        report[setting]['base_skew'] = base_skew
        report[setting]['performance'] = perf
        report[setting]['perf_skew'] = perf_skew

    print('Done', file=sys.stderr, flush=True)

    # Print experiment report
    print('Normal thread dir                :')
    for norm_dir in args.norm_dir:
        print('  {}'.format(norm_dir))
    print('OOT thread dir                   :')
    for oot_dir in args.oot_dir:
        print('  {}'.format(oot_dir))
    print('Number of normal posts           :', args.num_norm)
    print('Number of OOT posts              :', args.num_oot)
    print('OOT detection methods            :', ' '.join(args.method))
    print('Distance metrics                 :', ' '.join(args.metric))
    print('Number of posts in top list      :', args.num_top)
    print('Number of iterations             :', args.niter)

    print(len(settings), 'experiment(s)')
    print()

    for ii, setting in enumerate(sorted(report)):
        norm_dir, oot_dir, *rest = setting
        summary = report[setting]

        # Print experiment setting info
        if ii > 0 and ii % len(args.num_top) == 0:
            print()
        print('##### Experiment {} #####'.format(ii+1))
        print('  norm_dir =', norm_dir)
        print('  oot_dir =', oot_dir)
        txt = '  num_norm = {}, num_oot = {}, method = {}, metric = {}, '
        'num_top = {}'
        print(txt.format(*rest))

        # Print experiment result summary
        np.set_printoptions(precision=3, suppress=True,
                            formatter={'float': '{: 0.3f}'.format})
        print('  BASELINE:')
        print(' ', summary['baseline'], 'skew =', summary['base_skew'])
        print('  PERFORMANCE:')
        print(' ', summary['performance'], 'skew =', summary['base_skew'])
