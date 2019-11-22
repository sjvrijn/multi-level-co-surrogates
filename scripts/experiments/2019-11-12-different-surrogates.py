#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-11-12-different-surrogates.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples using different
surrogate models as backend instead of only using Kriging with Matern Kernel
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from itertools import product
import sys

from pyprojroot import here

import mf2 as mff

from experiments import Instance, create_model_error_grid

save_dir = here('files/2019-11-12-different-surrogates/')
save_dir.mkdir(parents=True, exist_ok=True)

cases = [
    mff.forrester,

    mff.Forrester(ndim=2),
    # mff.bohachevsky,
    # mff.booth,
    mff.branin,
    mff.currin,
    # mff.himmelblau,
    # mff.six_hump_camelback,

    mff.Forrester(ndim=4),
    mff.park91a,
    # mff.park91b,

    # mff.Forrester(ndim=6),
    mff.hartmann6,

    # mff.Forrester(ndim=8),
    mff.borehole,
]

surrogate_names = [
    'RandomForest',
    'ElasticNet',
    'SVM'
]

scaling_options = [
    'off',
    # 'on',
    # 'inverted',
    # 'regularized'
]

min_high, max_high = 2, 50
min_low, max_low = 3, 125
step = 1
num_reps = 50

instances = [Instance(h, l, r)
             for h, l, r in product(range(min_high, max_high + 1, step),
                                    range(min_low, max_low + 1, step),
                                    range(num_reps))
             if h < l]

full_cases = list(product(function_cases, surrogate_names, scaling_options))

if len(sys.argv) > 1:
    case_idx = int(sys.argv[1])
    full_cases = full_cases[case_idx:case_idx + 1]

for function_case, surr_name, scale in full_cases:
    mfbo_options = {'surrogate_name': surr_name, 'scaling': scale}
    create_model_error_grid(function_case, instances, mfbo_options, save_dir=save_dir)
