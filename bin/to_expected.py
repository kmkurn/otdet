#!/usr/bin/env python

import argparse

import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert result from PMF'
                                     ' to expected value')
    parser.add_argument('file', type=str,
                        help='Result DataFrame in HDF5 format')
    parser.add_argument('outfile', type=str,
                        help='Output file')
    parser.add_argument('--hdf-key', type=str, default='df',
                        help='Identifier in the HDF5 store')
    args = parser.parse_args()

    df = pd.read_hdf(args.file, args.hdf_key)
    data = np.array([])
    grouped = df.groupby(level=df.columns.names[:4], axis=1)
    columns = []
    for name, _ in grouped:
        columns.append(name)
        pmf = df[name].values
        supp = np.array(df[name].columns)
        expected = np.sum(supp*pmf, axis=1)
        data = np.concatenate((data, expected))

    index = df.index.copy()
    columns = pd.MultiIndex.from_tuples(columns)
    m, n = len(index), len(columns)
    data = data.reshape((n, m))
    df2 = pd.DataFrame(data.T, index=index, columns=columns)

    df2.to_hdf(args.outfile, args.hdf_key)
    print("Stored in HDF5 format with the name '{}'".format(args.hdf_key))
