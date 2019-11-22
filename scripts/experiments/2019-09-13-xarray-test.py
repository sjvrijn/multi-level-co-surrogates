#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-09-13-xarray-test.py: Experiment runner file for generating sample data to
test the new implementation using the xarray package
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from itertools import product, starmap

from pyprojroot import here

import mf2

from experiments import Instance, create_model_error_grid

save_dir = here('files/2019-09-xarray-test/')
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


# Some manually selected combinations to confirm that the resulting array is
# still somewhat sparse, but that everything can be indexed as intended
instances = list(starmap(Instance, [
    (2, 4, 0), (2, 7, 0), (2, 9, 0),
    (2, 4, 1), (2, 7, 1), (2, 9, 1),
    (4, 7, 0), (5, 7, 0), (5, 9, 0),
    (4, 7, 3), (5, 7, 3), (5, 9, 3),
    (6, 7, 0), (6, 9, 0), (7, 9, 0),
    (6, 7, 5), (6, 9, 5), (7, 9, 5),
]))

for case, kernel, scale in product(cases, kernels, scaling_options):
    mfbo_options = {'kernel': kernel, 'scaling': scale}
    create_model_error_grid(case, instances, mfbo_options, save_dir=save_dir)
