#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
20190822_experiment_highvlow.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from collections import namedtuple
from itertools import product

from pyprojroot import here

import multifidelityfunctions as mff

from .high_v_low_experiment import run

save_dir = here('./files/high_v_low/')

Case = namedtuple('Case', 'ndim func')

cases = [
    Case(1, mff.forrester),

    Case(2, mff.forrester),
    Case(2, mff.bohachevsky),
    Case(2, mff.booth),
    Case(2, mff.branin),
    Case(2, mff.currin),
    Case(2, mff.himmelblau),
    Case(2, mff.sixHumpCamelBack),

    Case(4, mff.forrester),
    Case(4, mff.park91a),
    Case(4, mff.park91b),

    Case(6, mff.forrester),
    Case(6, mff.hartmann6),

    Case(8, mff.forrester),
    Case(8, mff.borehole),
]

kernels = ['Matern_']
scaling_options = ['off']  # , 'on', 'inverted']  # , 'regularized']

min_high, max_high = 2, 50
min_low, max_low = 3, 125
step = 1
num_reps = 50

instances = [(h, l, r)
             for h, l, r in product(range(min_high, max_high + 1, step),
                                    range(min_low, max_low + 1, step),
                                    range(num_reps))
             if h < l]


run(cases, kernels, scaling_options, instances, save_dir=save_dir)
