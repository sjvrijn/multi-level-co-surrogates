# -*- coding: utf-8 -*-

"""
2019-11-01-extend-ratios.py: Experiment file to test hierarchical-model
accuracy on multi-fidelity functions for combinations of high/low numbers
of samples to allow for plotting results along equal ratios for different
numbers of high-fidelity samples.

Takes a single optional argument: the index of the case to run (0-14)
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import sys
from itertools import product

import numpy as np
from pyprojroot import here

import mf2

from experiments import Instance, create_model_error_grid


save_dir = here('files/2019-11-01-extend-ratios/')
save_dir.mkdir(parents=True, exist_ok=True)

cases = [
    mf2.forrester,

    mf2.Forrester(ndim=2),
    mf2.bohachevsky,
    mf2.booth,
    mf2.branin,
    mf2.currin,
    mf2.himmelblau,
    mf2.six_hump_camelback,

    mf2.Forrester(ndim=4),
    mf2.park91a,
    mf2.park91b,

    mf2.Forrester(ndim=6),
    mf2.hartmann6,

    mf2.Forrester(ndim=8),
    mf2.borehole,
]

kernels = ['Matern']
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
max_ratio = 20

n_highs = np.arange(min_high, max_high + 1, 10)
default_n_lows = np.arange(min_low, max_low + 1, step)
ratios = default_n_lows / n_highs[0]
ratios = ratios[ratios <= max_ratio]

instances = [Instance(h, int(ratio*h), rep)
             for h, ratio, rep in product(n_highs, ratios, range(num_reps))
             if int(ratio*h) > max_low]

extra_attributes = {'mf2_version': mf2.__version__}

if instances:

    if len(sys.argv) > 1:
        case_idx = int(sys.argv[1])
        cases = cases[case_idx:case_idx+1]

    for case, kernel, scale in product(cases, kernels, scaling_options):
        mfbo_options = {'kernel': kernel, 'scaling': scale}
        create_model_error_grid(case, instances, mfbo_options, save_dir=save_dir,
                                extra_attributes=extra_attributes)

else:
    print("No instances to run")
