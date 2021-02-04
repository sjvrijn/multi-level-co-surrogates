# -*- coding: utf-8 -*-

"""
2019-10-07-adjustables.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples, specifically
for the adjustable benchmark functions.
Adjustable parameter is expected as commandline argument.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import sys
from itertools import product

from pyprojroot import here

import mf2

from experiments import Instance, create_model_error_grid

save_dir = here('files/2019-10-07-adjustables/')
save_dir.mkdir(parents=True, exist_ok=True)

funcs = [
    mf2.adjustable.branin,
    mf2.adjustable.paciorek,
    mf2.adjustable.hartmann3,
    mf2.adjustable.trid,
]

cases = [
    f(float(x))
    for f in funcs
    for x in sys.argv[1:]
]

kernels = ['Matern']
scaling_options = [
    'off',
    # 'on',
    # 'inverted',
    # 'regularized'
]

min_high, max_high = 2, 50
min_low, max_low = 3, 125
step = 1
num_reps = 50

instances = [Instance(h, l, r)
             for h, l, r in product(range(min_high, max_high + 1, step),
                                    range(min_low, max_low + 1, step),
                                    range(num_reps))
             if h < l]

extra_attributes = {'mf2_version': mf2.__version__}

for case, kernel, scale in product(cases, kernels, scaling_options):
    if 'paciorek' in case.name.lower() and float(case.name[-3:]) == 0.0:
        continue  # In this case the functions are equal, leading to badly defined diff models

    mfbo_options = {'kernel': kernel, 'scaling': scale}
    create_model_error_grid(case, instances, mfbo_options, save_dir=save_dir,
                            extra_attributes=extra_attributes)
