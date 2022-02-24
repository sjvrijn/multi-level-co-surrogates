# -*- coding: utf-8 -*-

"""
2020-03-03-cv-bootstrap-resampling.py: runner file for bootstrap resampling
experiments: is the mse-plot gradient also visible in mse-plots based on
bootstrap-resampled DoE's? Additionally calculates MSE and R^2 based on
cross-validation principle by using only the left-out data
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


import argparse
from itertools import product

from pyprojroot import here

import mf2

from experiments import Instance, create_resampling_leftover_error_grid

save_dir = here('files/2020-03-03-cv-subsampling/')
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


def main(args):
    if args.idx:
        full_cases = [function_cases[idx] for idx in args.idx]
    else:
        full_cases = function_cases

    for DoE_high, DoE_low in args.doe_sizes:
        min_high, max_high = 2, DoE_high
        min_low, max_low = 3, DoE_low

        instances = [Instance(h, l, r)
                     for h, l, r in product(range(min_high, max_high),
                                            range(min_low, max_low),
                                            range(args.numreps))
                     if h < l]

        mfbo_options = {
            # 'surrogate_name': 'ElasticNet',
            'kernel': 'Matern',
            'scaling': 'off'
        }
        extra_attributes = {'mf2_version': mf2.__version__}

        for case in full_cases:
            create_resampling_leftover_error_grid(case, (DoE_high, DoE_low),
                                                  instances, mfbo_options, save_dir,
                                                  extra_attributes=extra_attributes)


def parse_doe_sizes(doe_sizes):
    """Parse ['30,75', '50,125'] into [(30, 75), (50, 125)]"""
    return [
        tuple(map(int, s.split(',')))
        for s in doe_sizes
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idx', nargs='*', type=int,
                        help='Indices of specific cases to run')
    parser.add_argument('--numreps', type=int, default=50,
                        help='Number of repetitions to perform. Default: 50')
    parser.add_argument('--doe-sizes', nargs='+', default=['30,75', '50,125'], metavar='N_HIGH,N_LOW',
                        help='Specify the initial DoE size(s). Default is both 30,75 and 50,125')
    args = parser.parse_args()
    args.doe_sizes = parse_doe_sizes(args.doe_sizes)

    main(args)
