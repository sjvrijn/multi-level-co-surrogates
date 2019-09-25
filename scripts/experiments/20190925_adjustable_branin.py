#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
20190822_highvlow.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from itertools import product

from pyprojroot import here

import multifidelityfunctions as mff

from experiments import Case, Instance, calculate_mse_grid

save_dir = here('files/high_v_low/')
save_dir.mkdir(parents=True, exist_ok=True)

cases = [
    Case(2, mff.adjustable_branin(0)),
    Case(2, mff.adjustable_branin(0.5)),
]

kernels = ['Matern_']
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
