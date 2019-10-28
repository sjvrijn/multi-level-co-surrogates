#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-09-10-plot-specifics.py: Experiment runner to generate plots of the search-
space and sample points for specific selected instances.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from itertools import product

from pyprojroot import here

import multifidelityfunctions as mff

from experiments import Case, Instance, plot_model_and_samples

save_dir = here('plots/2019-09-plot-specifics/')
save_dir.mkdir(parents=True, exist_ok=True)

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

kernels = ['Matern']
scaling_options = ['off']  # , 'on', 'inverted']  # , 'regularized']
instances = [Instance(h, l, r)
             for h, l, r in product(range(20, 51, 10),
                                    range(25, 126, 20),
                                    [1])
             if h < l]


for case, kernel, scale in product(cases, kernels, scaling_options):
    for instance in instances:
        plot_model_and_samples(case, kernel, scale, instance)
