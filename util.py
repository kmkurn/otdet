#!/usr/bin/env python

import glob
import os.path
import random


def pick_random(directory, k=None):
    """Pick randomly some files from a directory."""
    all_files = glob.glob(os.path.join(directory, '*'))
    random.shuffle(all_files)
    return all_files if k is None else all_files[:k]
