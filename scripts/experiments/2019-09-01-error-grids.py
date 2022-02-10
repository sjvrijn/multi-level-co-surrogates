# -*- coding: utf-8 -*-

"""
2019-09-01-error-grids.py: Experiment runner file for generating data for
many combinations of numbers of high- vs. low-fidelity samples
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import argparse
from itertools import product

from pyprojroot import here

import mf2

from experiments import Instance, create_model_error_grid

save_dir = here('files/2019-09-mse-nc/')
save_dir.mkdir(parents=True, exist_ok=True)

cases = [
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

kernels = ['Matern']
scaling_options = [
    'off',
    # 'on',
    # 'inverted',
    # 'regularized'
]

min_high, max_high = 2, 50
min_low, max_low = 3, 125
extra_attributes = {'mf2_version': mf2.__version__}

def main(args):
    full_cases = list(product(cases, kernels, scaling_options))

    if args.idx:
        full_cases = [full_cases[idx] for idx in args.idx]

    instances = [Instance(h, l, r)
                 for h, l, r in product(range(min_high, max_high + 1),
                                        range(min_low, max_low + 1),
                                        range(args.numreps))
                 if h < l]

    for case, kernel, scale in full_cases:
        mfbo_options = {'kernel': kernel, 'scaling': scale}
        create_model_error_grid(case, instances, mfbo_options, save_dir=save_dir,
                                extra_attributes=extra_attributes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idx', nargs='*', type=int,
                        help='Indices of specific cases to run')
    parser.add_argument('--numreps', type=int, default=50,
                        help='Number of repetitions to perform')
    args = parser.parse_args()

    main(args)