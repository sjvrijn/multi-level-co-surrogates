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

import multifidelityfunctions as mff

from .high_v_low_experiment import run

min_high = 2
min_low = 3
max_high = 50
max_low = 125
num_reps = 50
step = 1

Case = namedtuple('Case', 'ndim func func_name')

cases = [
    Case(1, mff.forrester, 'forrester'),

    Case(2, mff.forrester, 'forrester'),
    Case(2, mff.bohachevsky, 'bohachevsky'),
    Case(2, mff.booth, 'booth'),
    Case(2, mff.branin, 'branin'),
    Case(2, mff.currin, 'currin'),
    Case(2, mff.himmelblau, 'himmelblau'),
    Case(2, mff.sixHumpCamelBack, 'sixHumpCamelBack'),

    Case(4, mff.forrester, 'forrester'),
    Case(4, mff.park91a, 'park91a'),
    Case(4, mff.park91b, 'park91b'),

    Case(6, mff.forrester, 'forrester'),
    Case(6, mff.hartmann6, 'hartmann6'),

    Case(8, mff.forrester, 'forrester'),
    Case(8, mff.borehole, 'borehole'),
]

kernels = ['Matern_']
scaling_options = ['off']  # , 'on', 'inverted']  # , 'regularized']

instances = [(h, l, r)
             for h, l, r in product(range(min_high, max_high + 1, step),
                                    range(min_low, max_low + 1, step),
                                    range(num_reps))
             if h < l]


run(cases, kernels, scaling_options, instances)

