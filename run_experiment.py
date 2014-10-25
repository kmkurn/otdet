#!/usr/bin/env python

if __name__ == '__main__':
    import argparse
    from glob import glob
    import os.path

    from termcolor import cprint

    from otdet.detector import OOTDetector
    from otdet.evaluation import TopListEvaluator
    from otdet.util import pick

    parser = argparse.ArgumentParser(description='Run experiment '
                                     'with given settings')
    parser.add_argument('normdir', type=str,
                        help='Normal thread directory')
    parser.add_argument('ootdir', type=str,
                        help='Thread directory from which '
                        'OOT post will be taken')
    parser.add_argument('-m', '--num-norm', type=int, default=None,
                        help='Number of posts taken from '
                        'normal thread directory')
    parser.add_argument('-n', '--num-oot', type=int, default=None,
                        help='Number of posts taken from '
                        'another thread directory to be OOT posts')
    parser.add_argument('-a', '--method', type=str, nargs='+',
                        choices=['clust_dist', 'mean_comp', 'txt_comp_dist'],
                        help='OOT post detection method to use')
    parser.add_argument('-t', '--top', type=int, default=1,
                        help='Number of posts in top N list')
    parser.add_argument('--niter', type=int, default=1,
                        help='Number of iteration for each method')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Be verbose')
    parser.add_argument('-c', '--colorized', action='store_true',
                        help='Colorize output')
    args = parser.parse_args()

    for meth in args.method:
        print('##### Applying', meth, 'method #####')
        evaluator = TopListEvaluator(args.top)
        for ii in range(args.niter):
            # Obtain normal posts
            normfiles = pick(glob(os.path.join(args.normdir, '*.txt')),
                             k=args.num_norm, randomized=False)
            # Print obtained normal posts in verbose mode
            if args.verbose:
                print('>> Obtaining', len(normfiles), 'normal posts')
                for file in normfiles:
                    if args.colorized:
                        cprint(file, 'green')
                    else:
                        print(file)
            # Obtain OOT posts
            ootfiles = pick(glob(os.path.join(args.ootdir, '*.txt')),
                            k=args.num_oot)
            # Print obtained OOT posts in verbose mode
            if args.verbose:
                print('>> Obtaining', len(ootfiles), 'OOT posts:')
                for file in ootfiles:
                    if args.colorized:
                        cprint(file, 'red')
                    else:
                        print(file)
            # Combine them both
            files = normfiles + ootfiles
            truth = [False]*len(normfiles) + [True]*len(ootfiles)

            # Apply OOT post detection methods
            print('Iteration #{:02}:'.format(ii+1), end=' ')
            detector = OOTDetector(files)
            methodfunc = getattr(detector, meth)
            distances = methodfunc()
            print('OK')

            # Construct ranked list of OOT posts (1: most off-topic)
            # In case of tie, prioritize normal post (worst case)
            s = sorted(zip(files, distances, truth), key=lambda x: x[2])
            ranked = sorted(s, key=lambda x: x[1], reverse=True)

            # Print result in verbose mode
            if args.verbose:
                for i, (file, distance, oot) in enumerate(ranked):
                    if args.colorized:
                        txt = '#{:02} {} -> {}'.format(i+1, file, distance)
                        cprint(txt, 'red' if oot else 'green')
                    else:
                        sym = 'o' if oot else ' '
                        txt = '#{:02} {} {} -> {}'.format(i+1, sym, file,
                                                          distance)
                        print(txt)

            # Store ranked list for evaluation
            ranked = [(distance, oot) for _, distance, oot in ranked]
            evaluator.add_result(ranked)

        print('Baseline:', evaluator.baseline)
        print('Performance:', evaluator.get_performance)