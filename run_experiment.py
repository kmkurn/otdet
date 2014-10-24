#!/usr/bin/env python

if __name__ == '__main__':
    import argparse
    from glob import glob
    import os.path

    from termcolor import cprint

    from method import OffTopicDetector
    import util

    parser = argparse.ArgumentParser(description='Run experiment '
                                     'with given settings')
    parser.add_argument('normal_thread', type=str,
                        help='Normal thread directory')
    parser.add_argument('oot_thread', type=str,
                        help='Thread directory from which '
                        'OOT post will be taken')
    parser.add_argument('-m', type=int, default=None,
                        help='Number of posts taken from normal_thread')
    parser.add_argument('-n', type=int, default=None,
                        help='Number of posts taken from oot_thread')
    parser.add_argument('--method', type=str, default='clust_dist', nargs='*',
                        choices=['clust_dist', 'mean_comp', 'txt_comp_dist'],
                        help='OOT post detection method to use')
    args = parser.parse_args()

    # Obtain normal posts
    normfiles = util.pick(glob(os.path.join(args.normal_thread, '*')),
                          k=args.m, randomized=False)
    print('Obtaining', len(normfiles), 'normal posts')
    for file in normfiles:
        cprint(file, 'green')

    # Obtain OOT posts
    ootfiles = util.pick(glob(os.path.join(args.oot_thread, '*')), k=args.n)
    print('Obtaining', len(ootfiles), 'OOT posts:')
    for file in ootfiles:
        cprint(file, 'red')

    # Combine them both
    files = normfiles + ootfiles
    truth = [False]*len(normfiles) + [True]*len(ootfiles)

    # Apply OOT post detection methods
    detector = OffTopicDetector(files)
    for meth in args.method:
        print('\nApplying', meth, 'method...', end=' ')
        methodfunc = getattr(detector, meth)
        res = methodfunc()
        print('OK')

        # Construct ranked list of OOT posts (1: most off-topic)
        ranked = reversed(sorted(zip(files, res, truth), key=lambda x: x[1]))

        # Print result
        for i, (file, score, t) in enumerate(ranked):
            txt = '#{:02} {} -> {}'.format(i+1, file, score)
            cprint(txt, 'red' if t else 'green')
