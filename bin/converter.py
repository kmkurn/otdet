#!/usr/bin/env python

import argparse
import os.path

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HDF5 to other '
                                                 'format')
    parser.add_argument('filename', type=str, help='HDF5 filename')
    parser.add_argument('format', type=str, choices=['html', 'xlsx'],
                        help='Format to which file will be converted')
    parser.add_argument('-o', '--outdir', type=str, help='Output directory')
    parser.add_argument('--hdf-key', type=str, default='df',
                        help='Identifier of data frame in HDF5 file')
    args = parser.parse_args()

    df = pd.read_hdf(args.filename, args.hdf_key)
    filename, _ = os.path.splitext(os.path.basename(args.filename))
    if args.outdir is not None:
        outfile = os.path.join(args.outdir, filename)
    if args.format == 'html':
        df.to_html('{}.html'.format(outfile), float_format='{:0.3f}'.format)
    else:
        df.to_excel('{}.xlsx'.format(outfile), float_format='%0.3f',
                    engine='xlsxwriter')
