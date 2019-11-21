#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-09-25-adjustable-branin.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples, specifically
for the adjustable version of the Branin function.
Adjusting values A1 are expected as commandline arguments.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import sys
from itertools import product

from pyprojroot import here

import multifidelityfunctions as mff

from experiments import Instance, create_model_error_grid

save_dir = here('files/2019-09-25-adjustable-branin/')
save_dir.mkdir(parents=True, exist_ok=True)

cases = [
    mff.adjustable_branin(float(x))
    for x in sys.argv[1:]
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

instances = [Instance(h, l, r)
             for h, l, r in product(range(min_high, max_high + 1, step),
                                    range(min_low, max_low + 1, step),
                                    range(num_reps))
             if h < l]


for case, kernel, scale in product(cases, kernels, scaling_options):
    mfbo_options = {'kernel': kernel, 'scaling': scale}
    create_model_error_grid(case, instances, mfbo_options, save_dir=save_dir)
