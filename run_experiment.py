#!/usr/bin/env python

from glob import glob
import os.path

from otdet.detector import OOTDetector
from otdet.util import pick


def experiment(setting):
    """Do experiment with the specified setting."""
    norm_dir, oot_dir, num_norm, num_oot, method, metric, top = setting
    result = []
    for jj in range(args.niter):
        # Obtain normal posts
        normfiles = pick(glob(os.path.join(norm_dir, '*.txt')), k=num_norm,
                         randomized=False)
        # Obtain OOT posts
        ootfiles = pick(glob(os.path.join(oot_dir, '*.txt')), k=num_oot)
        # Combine them both
        files = normfiles + ootfiles
        truth = [False]*len(normfiles) + [True]*len(ootfiles)

        # Apply OOT post detection methods
        detector = OOTDetector(files)
        methodfunc = getattr(detector, method)
        distances = methodfunc(metric=metric)

        # Construct ranked list of OOT posts (1: most off-topic)
        # In case of tie, prioritize normal post (worst case)
        s = sorted(zip(files, distances, truth), key=lambda x: x[2])
        ranked = sorted(s, key=lambda x: x[1], reverse=True)

        # Append to result
        result.append((normfiles, ootfiles, ranked))
    return result


def evaluate(setting, result):
    """Evaluate an experiment result with the given setting."""
    num_norm, num_oot, num_top = setting[2], setting[3], setting[6]
    evaluator = TopListEvaluator(M=num_norm+num_oot, n=num_oot, N=num_top)
    trans_result = [[(distance, is_oot) for _, distance, is_oot in ranked]
                    for _, _, ranked in result]
    return evaluator.baseline, evaluator.get_performance(trans_result)


if __name__ == '__main__':
    import argparse
    from collections import defaultdict
    import itertools as it
    from multiprocessing import Pool
    import sys

    from termcolor import cprint

    from otdet.evaluation import TopListEvaluator

    parser = argparse.ArgumentParser(description='Run experiment '
                                     'with given settings')
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
    parser.add_argument('-v', '--verbose', action='count',
                        help='Be verbose')
    parser.add_argument('-c', '--colorized', action='store_true',
                        help='Colorize output')
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

    report = defaultdict(dict)
    for setting, result in zip(settings, results):
        # Store the report
        baseline, performance = evaluate(setting, result)
        report[setting]['iteration'] = result
        report[setting]['baseline'] = baseline
        report[setting]['performance'] = performance

    print('Done', file=sys.stderr, flush=True)

    # Print experiment report
    print('Normal thread dir                :')
    for norm_dir in args.norm_dir:
        print('  {}'.format(norm_dir))
    print('OOT thread dir                   :')
    for oot_dir in args.oot_dir:
        print('  {}'.format(oot_dir))
    if args.verbose >= 1:
        print('Number of normal posts           :', args.num_norm)
        print('Number of OOT posts              :', args.num_oot)
        print('OOT detection methods            :', ' '.join(args.method))
        print('Distance metrics                 :', ' '.join(args.metric))
        print('Number of posts in top list      :', args.num_top)
        print('Number of iterations             :', args.niter)

    print(len(settings), 'experiment(s)')
    print()

    for ii, setting in enumerate(sorted(report)):
        norm_dir, oot_dir, num_norm, *rest = setting

        # Preprocess num_norm
        if num_norm < 0:
            num_norm = len(glob(os.path.join(norm_dir, '*.txt')))

        # Print experiment setting info
        print('##### Experiment {} #####'.format(ii+1))
        print('  norm_dir =', norm_dir)
        print('  oot_dir =', oot_dir)
        txt = '  m = {}, n = {}, method = {}, metric = {}, t = {}'
        print(txt.format(num_norm, *rest))

        # Print obtained normal posts in very verbose mode
        if args.verbose >= 2:
            for normfiles, ootfiles, ranked in report[setting]['iteration']:
                print('    >> Obtaining', len(normfiles), 'normal posts')
                for file in normfiles:
                    txt = '    {}'.format(file)
                    if args.colorized:
                        cprint(txt, 'green')
                    else:
                        print(txt)

                print('    >> Obtaining', len(ootfiles), 'OOT posts:')
                for file in ootfiles:
                    txt = '    {}'.format(file)
                    if args.colorized:
                        cprint(txt, 'red')
                    else:
                        print(txt)

                print('    >> Result')
                for i, (file, distance, oot) in enumerate(ranked):
                    if args.colorized:
                        txt = '    #{:02} {} -> {}'.format(i+1, file, distance)
                        cprint(txt, 'red' if oot else 'green')
                    else:
                        sym = 'o' if oot else ' '
                        txt = '    #{:02} {} {} -> {}'.format(i+1, sym, file,
                                                              distance)
                        print(txt)

        baseline = ['{:.6f}'.format(p)
                    for p in report[setting]['baseline']]
        performance = ['{:.6f}'.format(p)
                       for p in report[setting]['performance']]
        print('  BASELINE:', '  '.join(baseline))
        print('  PERFORMANCE:', '  '.join(performance))
