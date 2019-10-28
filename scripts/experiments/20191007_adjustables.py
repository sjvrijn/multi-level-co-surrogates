#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
20190925_adjustable_branin.py: Experiment runner file for generating data for
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

from experiments import Case, Instance, calculate_mse_grid

save_dir = here('files/adjustables/')
save_dir.mkdir(parents=True, exist_ok=True)

funcs = [
    (2, mff.adjustable_branin),
    (2, mff.adjustable_paciorek),
    (3, mff.adjustable_hartmann3),
    (10, mff.adjustable_trid),
]

cases = [
    Case(ndim, f(float(x)))
    for ndim, f in funcs
    for x in sys.argv[1:]
]

kernels = ['Matern']
scaling_options = ['off']  # , 'on', 'inverted']  # , 'regularized']

min_high, max_high = 2, 50
min_low, max_low = 3, 125
step = 1
num_reps = 50

instances = [Instance(h, l, r)
             for h, l, r in product(range(min_high, max_high + 1, step),
                                    range(min_low, max_low + 1, step),
                                    range(num_reps))
             if h < l]


calculate_mse_grid(cases, kernels, scaling_options, instances, save_dir=save_dir)
