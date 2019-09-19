#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
20190910_plot_specifics.py: Experiment runner to generate plots of the search-
space and sample points for specific selected instances.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from collections import namedtuple
from itertools import product

from pyprojroot import here

import multifidelityfunctions as mff

from experiments import plot_model_and_samples

save_dir = here('plots/plot_specifics/')
save_dir.mkdir(parents=True, exist_ok=True)

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
