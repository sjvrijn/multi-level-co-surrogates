#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-03-04-cv-adjustables-resampling-edd.py: Experiment runner file for generating
data on bootstrap resampling from a single DoE with added 'cross-validation'
error calculation included for adjustable functions
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import subprocess
from itertools import product
import numpy as np

seed_offsets = range(0, 3)
cases = range(1, 21, 2)
scales = np.round(np.linspace(0.0, 1.0, 6)[1:], 2)

for seed_offset, scale, case_idx in product(seed_offsets, scales, cases):
    subprocess.run(f'python3 2020-03-04-cv-adjustables-resampling.py {case_idx} {scale} {seed_offset}', shell=True)
