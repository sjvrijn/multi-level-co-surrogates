#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-11-01-extend-ratios.py: Experiment file to test hierarchical-model
accuracy on multi-fidelity functions for combinations of high/low numbers
of samples to allow for plotting results along equal ratios for different
numbers of high-fidelity samples.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from itertools import product

import numpy as np
from pyprojroot import here

import multifidelityfunctions as mff

from experiments import Case, Instance, calculate_mse_grid


save_dir = here('files/2019-11-01-extend-ratios/')
save_dir.mkdir(parents=True, exist_ok=True)

cases = [
    Case(1, mff.forrester),

    Case(2, mff.forrester),
    Case(2, mff.bohachevsky),
    Case(2, mff.booth),
    Case(2, mff.branin),
    Case(2, mff.currin),
    Case(2, mff.himmelblau),
    Case(2, mff.six_hump_camelback),

    Case(4, mff.forrester),
    Case(4, mff.park91a),
    Case(4, mff.park91b),

    Case(6, mff.forrester),
    Case(6, mff.hartmann6),

    Case(8, mff.forrester),
    Case(8, mff.borehole),
]

kernels = ['Matern']
scaling_options = ['off']  # , 'on', 'inverted']  # , 'regularized']

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
             for h, ratio, rep in product(n_highs, ratios, range(num_reps))]

calculate_mse_grid(cases, kernels, scaling_options, instances, save_dir=save_dir)
