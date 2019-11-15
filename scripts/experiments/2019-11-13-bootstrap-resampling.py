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

from pyprojroot import here

import multifidelityfunctions as mff

from experiments import Case, Instance, create_resampling_error_grid

save_dir = here('files/2019-11-15-subsampling/')
save_dir.mkdir(parents=True, exist_ok=True)

function_cases = [
    Case(1, mff.forrester),

    Case(2, mff.forrester),
    Case(2, mff.bohachevsky),
    Case(2, mff.booth),
    Case(2, mff.branin),
    Case(2, mff.currin),
    Case(2, mff.himmelblau),
    Case(2, mff.six_hump_camelback),

    Case(4, mff.forrester),
    Case(4, mff.park91a),
    Case(4, mff.park91b),

    Case(6, mff.forrester),
    Case(6, mff.hartmann6),

    Case(8, mff.forrester),
    Case(8, mff.borehole),
]



min_high, max_high = 2, 50
min_low, max_low = 3, 125
step = 1
num_reps = 3

DoE_high, DoE_low = 5, 10
instances = [Instance(h, l, r)
             for h, l, r in product(range(min_high, DoE_high),
                                    range(min_low, DoE_low+1),
                                    range(num_reps))
             if h < l]


mfbo_options = {'kernel': 'Matern', 'scaling': 'off'}

create_resampling_error_grid(function_cases[0], (DoE_high, DoE_low), instances, mfbo_options, save_dir)
