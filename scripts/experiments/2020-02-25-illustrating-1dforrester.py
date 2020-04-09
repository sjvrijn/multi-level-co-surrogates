#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-02-25-illustrating-1dforrester.py: runner file to run a few
examples of the 1D forrester function to investigate why there is
a jump in MSE when n_low / n_high == 1.5.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from itertools import product
import sys

from pyprojroot import here

import mf2

from experiments import Instance, create_model_error_grid

#save_dir = here('files/2019-08-mse-nc/')
save_dir = here('files/2020-02-25-illustrating-1dforrester/')
save_dir.mkdir(parents=True, exist_ok=True)

cases = [
    mf2.forrester,

    #mf2.Forrester(ndim=2),
    #mf2.bohachevsky,
    #mf2.booth,
    #mf2.branin,
    #mf2.currin,
    #mf2.himmelblau,
    #mf2.six_hump_camelback,

    #mf2.Forrester(ndim=4),
    #mf2.park91a,
    #mf2.park91b,

    #mf2.Forrester(ndim=6),
    #mf2.hartmann6,

    #mf2.Forrester(ndim=8),
    #mf2.borehole,
]

kernels = ['Matern']
scaling_options = [
    'off',
    # 'on',
    # 'inverted',
    # 'regularized'
]

min_high, max_high = 10, 30
#min_low, max_low = 3, 125
#step = 1
#num_reps = 50

#instances = [Instance(h, l, r)
#             for h, l, r in product(range(min_high, max_high + 1, step),
#                                    range(min_low, max_low + 1, step),
#                                    range(num_reps))
#             if h < l]
ratio = 1.5
instances = [Instance(h, l, 1)
             for h in range(min_high, max_high+1)
             for l in range(int(h*ratio)-1, int(h*ratio)+2)]

#print(instances)
#import sys
#sys.exit(0)

full_cases = list(product(cases, kernels, scaling_options))
extra_attributes = {'mf2_version': mf2.__version__}

if len(sys.argv) > 1:
    case_idx = int(sys.argv[1])
    full_cases = full_cases[case_idx:case_idx+1]


for case, kernel, scale in full_cases:
    mfbo_options = {'kernel': kernel, 'scaling': scale}
    create_model_error_grid(case, instances, mfbo_options, save_dir=save_dir,
                            plot_1d=True, extra_attributes=extra_attributes)
