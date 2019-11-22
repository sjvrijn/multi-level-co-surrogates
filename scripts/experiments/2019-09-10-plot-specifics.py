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

import mf2

from experiments import Instance, plot_model_and_samples

save_dir = here('plots/2019-09-plot-specifics/')
save_dir.mkdir(parents=True, exist_ok=True)

cases = [
    # mf2.forrester,

    # mf2.Forrester(ndim=2),
    # mf2.bohachevsky,
    # mf2.booth,
    mf2.branin,
    # mf2.currin,
    # mf2.himmelblau,
    # mf2.sixHumpCamelBack,
]

kernels = ['Matern']
scaling_options = [
    'off',
    # 'on',
    # 'inverted',
    # 'regularized'
]
instances = [Instance(h, l, r)
             for h, l, r in product(range(20, 51, 10),
                                    range(25, 126, 20),
                                    [1])
             if h < l]


for case, kernel, scale in product(cases, kernels, scaling_options):
    for instance in instances:
        plot_model_and_samples(case, kernel, scale, instance)
