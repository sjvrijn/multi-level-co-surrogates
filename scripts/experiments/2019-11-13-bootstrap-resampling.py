#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-11-13-bootstrap-resampling.py: runner file for bootstrap resampling
experiments: is the mse-plot gradient also visible in mse-plots based on
bootstrap-resampled DoE's?
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


from itertools import product
import sys

from pyprojroot import here

import mf2

from experiments import Instance, create_resampling_error_grid

save_dir = here('files/2019-11-15-subsampling/')
save_dir.mkdir(parents=True, exist_ok=True)

function_cases = [
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


if len(sys.argv) > 1:
    case_idx = int(sys.argv[1])
    function_cases = function_cases[case_idx:case_idx+1]
    scale = float(sys.argv[2])
else:
    scale = 1


DoE_high, DoE_low = 50, 125
final_num_reps = 15

min_high, max_high = 2, int(DoE_high * scale)
min_low, max_low = 3, int(DoE_low * scale)
step = 1
num_reps = max(1, int(final_num_reps * scale))

instances = [Instance(h, l, r)
             for h, l, r in product(range(min_high, max_high),
                                    range(min_low, max_low),
                                    range(num_reps))
             if h < l]


mfbo_options = {
    'surrogate_name': 'ElasticNet',
    # 'kernel': 'Matern',
    'scaling': 'off'}
extra_attributes = {'mf2_version': mf2.__version__}


for case in function_cases:
    create_resampling_error_grid(case, (DoE_high, DoE_low), instances, mfbo_options,
                                 save_dir, extra_attributes=extra_attributes)
