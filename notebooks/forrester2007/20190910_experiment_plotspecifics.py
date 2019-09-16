#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Filename.py: << A short summary docstring about this file >>
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from collections import namedtuple
from itertools import product

import multifidelityfunctions as mff

from high_v_low_experiment import plot_model_and_samples


Case = namedtuple('Case', 'ndim func')
Instance = namedtuple('Instance', 'high low rep')

cases = [
    # Case(1, mff.forrester),

    # Case(2, mff.forrester),
    # Case(2, mff.bohachevsky),
    # Case(2, mff.booth),
    Case(2, mff.branin),
    # Case(2, mff.currin),
    # Case(2, mff.himmelblau),
    # Case(2, mff.sixHumpCamelBack),

    # Case(4, mff.forrester),
    # Case(4, mff.park91a),
    # Case(4, mff.park91b),

    # Case(6, mff.forrester),
    # Case(6, mff.hartmann6),

    # Case(8, mff.forrester),
    # Case(8, mff.borehole),
]

kernels = ['Matern_']
scaling_options = ['off']  # , 'on', 'inverted']  # , 'regularized']
instances = [Instance(h, l, r)
             for h, l, r in product(range(20, 51, 10),
                                    range(25, 126, 20),
                                    [1])
             if h < l]


for case, kernel, scale in product(cases, kernels, scaling_options):
    for instance in instances:
        plot_model_and_samples(case, kernel, scale, instance)
