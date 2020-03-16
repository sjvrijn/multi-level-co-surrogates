#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-03-04-cv-adjustables-resampling.py: runner file for bootstrap resampling
experiments: is the mse-plot gradient also visible in mse-plots based on
bootstrap-resampled DoE's? Uses adjustable correlation functions to sample a
spread of possible angle values. Additionally calculates MSE and R^2 based on
cross-validation principle by using only the left-out data
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


from itertools import product
import sys

from pyprojroot import here

import mf2

from experiments import Instance, create_resampling_leftover_error_grid

save_dir = here('files/2020-03-04-cv-adjustables-subsampling/')
save_dir.mkdir(parents=True, exist_ok=True)

function_cases = [
    *[mf2.adjustable.branin(a1)    for a1 in [0.00, 0.05, 0.25]],
    *[mf2.adjustable.paciorek(a2)  for a2 in [0.05, 0.10, 0.15, 0.20, 0.25]],
    *[mf2.adjustable.hartmann3(a3) for a3 in [0.20, 0.25, 0.30, 0.35, 0.40]],
    *[mf2.adjustable.trid(a4)      for a4 in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]],
]


scale = 1
seed_offset = 0

if len(sys.argv) >= 3:
    case_idx = int(sys.argv[1])
    function_cases = function_cases[case_idx:case_idx+1]
    scale = float(sys.argv[2])

if len(sys.argv) == 4:
    seed_offset = int(sys.argv[3])


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
    # 'surrogate_name': 'ElasticNet',
    'kernel': 'Matern',
    'scaling': 'off'
}


for case in function_cases:
    create_resampling_leftover_error_grid(case, (DoE_high, DoE_low),
                                          instances, mfbo_options, save_dir,
                                          seed_offset=seed_offset)
