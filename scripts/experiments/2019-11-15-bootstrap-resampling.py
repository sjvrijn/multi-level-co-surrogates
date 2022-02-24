# -*- coding: utf-8 -*-

"""
2019-11-15-bootstrap-resampling.py: runner file for bootstrap resampling
experiments: is the mse-plot gradient also visible in mse-plots based on
bootstrap-resampled DoE's?

Takes two optional arguments:
 - the index of the case to run (0-14)
 - the 'scale' [0-1] of how much to sample
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


import argparse
from itertools import product

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

DoE_high, DoE_low = 50, 125

min_high, max_high = 2, DoE_high
min_low, max_low = 3, DoE_low

mfbo_options = {
    'kernel': 'Matern',
    'scaling': 'off'
}
extra_attributes = {'mf2_version': mf2.__version__}


def main(args):

    instances = [Instance(h, l, r)
                 for h, l, r in product(range(min_high, max_high),
                                        range(min_low, max_low),
                                        range(args.numreps))
                 if h < l]
    if not instances:
        return

    cases = [function_cases[idx] for idx in args.idx] if args.idx else function_cases

    for case in cases:
        create_resampling_error_grid(case, (DoE_high, DoE_low), instances, mfbo_options,
                                     save_dir, extra_attributes=extra_attributes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idx', nargs='*', type=int,
                        help='Indices of specific cases to run')
    parser.add_argument('--numreps', type=int, default=50,
                        help='Number of repetitions to perform. Default: 50')
    args = parser.parse_args()

    main(args)
