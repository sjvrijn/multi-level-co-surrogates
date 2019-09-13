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

from high_v_low_experiment import run


Case = namedtuple('Case', 'ndim func func_name')

cases = [
    Case(1, mff.forrester, 'forrester'),

    # Case(2, mff.forrester, 'forrester'),
    # Case(2, mff.bohachevsky, 'bohachevsky'),
    # Case(2, mff.booth, 'booth'),
    # Case(2, mff.branin, 'branin'),
    # Case(2, mff.currin, 'currin'),
    # Case(2, mff.himmelblau, 'himmelblau'),
    # Case(2, mff.sixHumpCamelBack, 'sixHumpCamelBack'),
    #
    # Case(4, mff.forrester, 'forrester'),
    # Case(4, mff.park91a, 'park91a'),
    # Case(4, mff.park91b, 'park91b'),
    #
    # Case(6, mff.forrester, 'forrester'),
    # Case(6, mff.hartmann6, 'hartmann6'),
    #
    # Case(8, mff.forrester, 'forrester'),
    # Case(8, mff.borehole, 'borehole'),
]

kernels = ['Matern_']
scaling_options = ['off']  # , 'on', 'inverted']  # , 'regularized']


# Some manually selected combinations to confirm that the resulting array is
# still somewhat sparse, but that everything can be indexed as intended
instances = [(2, 4, 0), (2, 7, 0), (2, 9, 0),
             (2, 4, 1), (2, 7, 1), (2, 9, 1),
             (4, 7, 0), (5, 7, 0), (5, 9, 0),
             (4, 7, 3), (5, 7, 3), (5, 9, 3),
             (6, 7, 0), (6, 9, 0), (7, 9, 0),
             (6, 7, 5), (6, 9, 5), (7, 9, 5),
             ]


a, b, c = run(cases, kernels, scaling_options, instances)
# print(a, end=f'\n{"="*30}\n')
# print(b, end=f'\n{"="*30}\n')
# print(c, end=f'\n{"="*30}\n')

