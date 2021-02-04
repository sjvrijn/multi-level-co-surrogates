# -*- coding: utf-8 -*-

"""
2019-11-12-different-surrogates.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples using different
surrogate models as backend instead of only using Kriging with Matern Kernel
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from itertools import product
import sys

from pyprojroot import here

import mf2

from experiments import Instance, create_model_error_grid

save_dir = here('files/2019-11-12-different-surrogates/')
save_dir.mkdir(parents=True, exist_ok=True)

cases = [
    mf2.forrester,

    # mf2.Forrester(ndim=2),
    mf2.bohachevsky,
    mf2.booth,
    mf2.branin,
    mf2.currin,
    mf2.himmelblau,
    mf2.six_hump_camelback,

    # mf2.Forrester(ndim=4),
    mf2.park91a,
    mf2.park91b,

    # mf2.Forrester(ndim=6),
    mf2.hartmann6,

    # mf2.Forrester(ndim=8),
    mf2.borehole,
]

surrogate_names = [
    'RandomForest',
    'ElasticNet',
    'SVM',
    'RBF',
]

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

full_cases = list(product(cases, surrogate_names, scaling_options))
extra_attributes = {'mf2_version': mf2.__version__}

if len(sys.argv) > 1:
    case_idx = int(sys.argv[1])
    full_cases = full_cases[case_idx:case_idx + 1]

for function, surr_name, scale in full_cases:
    mfbo_options = {'surrogate_name': surr_name, 'scaling': scale}
    create_model_error_grid(function, instances, mfbo_options, save_dir=save_dir,
                            extra_attributes=extra_attributes)
