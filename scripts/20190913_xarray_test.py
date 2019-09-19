#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
20190913_xarray_test.py: Experiment runner file for generating sample data to
test the new implementation using the xarray package
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from collections import namedtuple

from pyprojroot import here

import multifidelityfunctions as mff

from experiments import calculate_mse_grid

save_dir = here('files/Xarray_test/')
save_dir.mkdir(parents=True, exist_ok=True)

Case = namedtuple('Case', 'ndim func')

cases = [
    Case(1, mff.forrester),

    # Case(2, mff.forrester),
    # Case(2, mff.bohachevsky),
    # Case(2, mff.booth),
    # Case(2, mff.branin),
    # Case(2, mff.currin),
    # Case(2, mff.himmelblau),
    # Case(2, mff.sixHumpCamelBack),
    #
    # Case(4, mff.forrester),
    # Case(4, mff.park91a),
    # Case(4, mff.park91b),
    #
    # Case(6, mff.forrester),
    # Case(6, mff.hartmann6),
    #
    # Case(8, mff.forrester),
    # Case(8, mff.borehole),
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


calculate_mse_grid(cases, kernels, scaling_options, instances, save_dir=save_dir)
