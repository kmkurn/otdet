#!/usr/bin/env python

if __name__ == '__main__':
    import argparse
    from collections import defaultdict
    from glob import glob
    import itertools as it
    import os.path
    import sys

    from termcolor import cprint

    from otdet.detector import OOTDetector
    from otdet.evaluation import TopListEvaluator
    from otdet.util import pick

    parser = argparse.ArgumentParser(description='Run experiment '
                                     'with given settings')
    parser.add_argument('-nd', '--normdir', type=str, nargs='+', required=True,
                        help='Normal thread directory')
    parser.add_argument('-od', '--ootdir', type=str, nargs='+', required=True,
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
    parser.add_argument('-t', '--top', type=int, nargs='+', required=True,
                        help='Number of posts in top N list')
    parser.add_argument('--niter', type=int, default=1,
                        help='Number of iteration for each method')
    parser.add_argument('-v', '--verbose', action='count',
                        help='Be verbose')
    parser.add_argument('-c', '--colorized', action='store_true',
                        help='Colorize output')
    args = parser.parse_args()

    # Experiment settings
    expr_settings = list(it.product(args.normdir, args.ootdir, args.num_norm,
                                    args.num_oot, args.method, args.metric,
                                    args.top))

    # Progress-related variables
    total_ops = len(expr_settings) * args.niter
    checkpoint, step = 0.1, 0
    # progress, chunk, step = 1, total_ops / 100, 0

    report = defaultdict(dict)
    for ii, setting in enumerate(expr_settings):
        normdir, ootdir, num_norm, num_oot, method, metric, top = setting
        # Begin experiment
        evaluator = TopListEvaluator(top)
        report[setting]['iteration'] = []
        for jj in range(args.niter):
            # Obtain normal posts
            normfiles = pick(glob(os.path.join(normdir, '*.txt')), k=num_norm,
                             randomized=False)
            # Obtain OOT posts
            ootfiles = pick(glob(os.path.join(ootdir, '*.txt')), k=num_oot)
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

            # Put all to report
            tup = (normfiles, ootfiles, ranked)
            report[setting]['iteration'].append(tup)

            # Store ranked list for evaluation
            ranked = [(distance, oot) for _, distance, oot in ranked]
            evaluator.add_result(ranked)

            # Print progress to stderr
            print('.', end='', file=sys.stderr, flush=True)
            if total_ops > 100:
                step += 1
                progress = step / total_ops
                if progress >= checkpoint:
                    print('{:.0f}%'.format(progress*100), end='',
                          file=sys.stderr, flush=True)
                    while progress >= checkpoint:
                        checkpoint += 0.1

        report[setting]['baseline'] = evaluator.baseline
        report[setting]['performance'] = evaluator.get_performance

    print(' Done', file=sys.stderr, flush=True)

    # Print experiment report
    print('Normal thread dir                :')
    for normdir in args.normdir:
        print('  {}'.format(normdir))
    print('OOT thread dir                   :')
    for ootdir in args.ootdir:
        print('  {}'.format(ootdir))
    if args.verbose >= 1:
        print('Number of normal posts           :', args.num_norm)
        print('Number of OOT posts              :', args.num_oot)
        print('OOT detection methods            :', ' '.join(args.method))
        print('Distance metrics                 :', ' '.join(args.metric))
        print('Number of posts in top list      :', args.top)
        print('Number of iterations             :', args.niter)

    print(len(expr_settings), 'experiment(s)')
    print()

    for ii, setting in enumerate(sorted(report)):
        normdir, ootdir, num_norm, *rest = setting

        # Preprocess num_norm
        if num_norm < 0:
            num_norm = len(glob(os.path.join(args.normdir, '*.txt')))

        # Print experiment setting info
        print('##### Experiment {} #####'.format(ii+1))
        txt = '  normdir = {}\n ootdir = {}'
        print(txt.format(normdir, ootdir))
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
        print('  Baseline:', '  '.join(baseline))
        print('  Performance:', '  '.join(performance))
