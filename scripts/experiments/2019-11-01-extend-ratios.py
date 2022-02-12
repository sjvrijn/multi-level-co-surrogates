# -*- coding: utf-8 -*-

"""
2019-11-01-extend-ratios.py: Experiment file to test hierarchical-model
accuracy on multi-fidelity functions for combinations of high/low numbers
of samples to allow for plotting results along equal ratios for different
numbers of high-fidelity samples.

Takes a single optional argument: the index of the case to run (0-14)
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import argparse
from itertools import product

import numpy as np
from pyprojroot import here

import mf2

from experiments import Instance, create_model_error_grid


save_dir = here('files/2019-11-01-extend-ratios/')
save_dir.mkdir(parents=True, exist_ok=True)

func_cases = [
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


def main(args):
    n_highs = np.arange(min_high, max_high + 1, 10)
    default_n_lows = np.arange(min_low, max_low + 1)
    ratios = default_n_lows / n_highs[0]
    ratios = ratios[ratios <= args.maxratio]

    instances = [Instance(h, int(ratio*h), rep)
                 for h, ratio, rep in product(n_highs, ratios, range(args.numreps))
                 if int(ratio*h) > max_low]

    extra_attributes = {'mf2_version': mf2.__version__}

    if not instances:
        print("No instances to run")
        return

    cases = [func_cases[idx] for idx in args.idx] if args.idx else func_cases

    for case, kernel, scale in product(cases, kernels, scaling_options):
        mfbo_options = {'kernel': kernel, 'scaling': scale}
        create_model_error_grid(case, instances, mfbo_options, save_dir=save_dir,
                                extra_attributes=extra_attributes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idx', nargs='*', type=int,
                        help='Indices of specific cases to run')
    parser.add_argument('--numreps', type=int, default=50,
                        help='Number of repetitions to perform. Default: 50')
    parser.add_argument('--maxratio', type=int, default=20,
                        help='Maximum ratio. Default: 20')
    args = parser.parse_args()

    main(args)
