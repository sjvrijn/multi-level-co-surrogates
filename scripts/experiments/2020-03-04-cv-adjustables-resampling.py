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


import argparse
from itertools import product

from pyprojroot import here

import mf2

from experiments import Instance, create_resampling_leftover_error_grid

save_dir = here('files/2020-03-04-cv-adjustables-subsampling/')
save_dir.mkdir(parents=True, exist_ok=True)

# Parameters selected to represent a roughly uniform spread of gradient angles
function_cases = [
    *[mf2.adjustable.branin(a1)    for a1 in [0.00, 0.05, 0.25]],
    *[mf2.adjustable.paciorek(a2)  for a2 in [0.05, 0.10, 0.15, 0.20, 0.25]],
    *[mf2.adjustable.hartmann3(a3) for a3 in [0.20, 0.25, 0.30, 0.35, 0.40]],
    *[mf2.adjustable.trid(a4)      for a4 in [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]],
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

        for case, offset in product(full_cases, args.seeds):
            create_resampling_leftover_error_grid(case, (DoE_high, DoE_low),
                                                  instances, mfbo_options, save_dir,
                                                  seed_offset=offset,
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
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Seed offsets to run with. Default: 0 for DoE size 30,75, 0...4 for 50,125')
    parser.add_argument('--doe-sizes', nargs='+', default=None, metavar='N_HIGH,N_LOW',
                        help='Specify the initial DoE size(s). Default is both 30,75 and 50,125')
    args = parser.parse_args()

    if args.doe_sizes is None:
        args.doe_sizes = ['30,75', '50,125']
        if args.seeds is None:
            args.seeds = [[0], range(5)]
        else:
            args.seeds = [args.seeds] * len(args.doe_sizes)
    elif args.seeds is None:
        args.seeds = [range(5)] * len(args.doe_sizes)
    else:
        args.seeds = [args.seeds] * len(args.doe_sizes)

    args.doe_sizes = parse_doe_sizes(args.doe_sizes)

    main(args)
