#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-11-05-simple-mfbo.py: Experiment file for running simple multi-fidelity
bayesian optimizers
"""

import argparse
from itertools import product
from warnings import warn, simplefilter

import mf2
import numpy as np
from pyprojroot import here
from tqdm import TqdmWarning

import multiLevelCoSurrogates as mlcs

save_dir = here('files/2020-11-05-simple-mfbo/', warn=False)
save_dir.mkdir(parents=True, exist_ok=True)

FOLDER_NAME_TEMPLATE = '{func_name}-{name}-c{cost_ratio:f}-b{budget:d}-i{idx:d}'

FIDELITY_SELECTORS = {
    'fixed': mlcs.FidelitySelection.FIXED,
    'naive': mlcs.FidelitySelection.NAIVE_EG,
    'proto-eg': mlcs.FidelitySelection.PROTO_EG,
}

FUNCTIONS = [
    mf2.bohachevsky,  # 0
    mf2.booth,  # 1
    mf2.branin,  # 2
    mf2.invert(mf2.currin),  # 3
    mf2.himmelblau,  # 4
    mf2.six_hump_camelback,  # 5

    mf2.park91a,  # 6
    mf2.park91b,  # 7

    mf2.hartmann6,  # 8

    mf2.borehole,  # 9

    *[mf2.adjustable.branin(a) for a in np.round(np.linspace(0, 1, 21),2)],       # 10-30
    *[mf2.adjustable.paciorek(a) for a in np.round(np.linspace(0.05, 1, 20),2)],  # 31-50
    *[mf2.adjustable.hartmann3(a) for a in np.round(np.linspace(0, 1, 21),2)],    # 51-71
    *[mf2.adjustable.trid(a) for a in np.round(np.linspace(0, 1, 21),2)],         # 72-92
]


def do_run(benchmark_func, fidelity_selector, kwargs, force_rerun=False):
    run_save_dir = kwargs['run_save_dir']
    run_save_dir.mkdir(parents=True, exist_ok=True)
    if force_rerun:
        for file in run_save_dir.iterdir():
            file.unlink()
    elif list(run_save_dir.iterdir()):
        return  # directory already contains files: don't overwrite

    optimizer = mlcs.Optimizer(
        func=benchmark_func,
        fid_selection_method=fidelity_selector,
        **kwargs,
    )
    return optimizer.iterate()


def main(args):
    import sklearn
    simplefilter("ignore", category=FutureWarning)
    simplefilter("ignore", category=sklearn.exceptions.ConvergenceWarning)
    simplefilter("ignore", category=mlcs.UnhelpfulTauWarning)
    simplefilter("ignore", category=mlcs.LowHighFidSamplesWarning)
    simplefilter("ignore", category=TqdmWarning)

    functions = [FUNCTIONS[f_idx] for f_idx in args.idx]

    kwargs = {
        'init_budget': args.budget,
        'cost_ratio': args.cost_ratio,
        'doe_n_high': 5,
        'doe_n_low': 10,
        'num_reps': args.nreps,
        'use_x_opt': args.shortcut,
    }

    for func in functions:
        print(func.name)

        for idx, cost_ratio, experiment in product(range(args.niters), args.cost_ratio, args.experiment):
            kwargs['seed_offset'] = idx
            kwargs['cost_ratio'] = cost_ratio
            kwargs['run_save_dir'] = save_dir / FOLDER_NAME_TEMPLATE.format(
                func_name=func.name,
                name=experiment,
                cost_ratio=cost_ratio,
                budget=args.budget,
                idx=idx,
            )
            print(f'    {experiment} c{cost_ratio} b{args.budget} i{idx}...')
            do_run(func, FIDELITY_SELECTORS[experiment], kwargs, args.force_rerun)


if __name__ == '__main__':

    cost_ratios = [0.1, 0.2, 0.25, 0.5]
    experiments = ['fixed', 'naive', 'proto-eg']

    parser = argparse.ArgumentParser()
    parser.add_argument('idx', type=int, default=range(len(FUNCTIONS)), nargs='*',
                        help=f'Experiment indices [0-{len(FUNCTIONS) - 1}] to run. Default: all')
    parser.add_argument('-e', '--experiment', type=str, default=experiments, nargs='*',
                        help=f'Experiment function to run. Options: {", ".join(experiments)}. Default: all')
    parser.add_argument('--nreps', type=int, default=50,
                        help='number of independent repetitions to perform for the error grid')
    parser.add_argument('--niters', type=int, default=5,
                        help='number of independent iterations of the experiment to perform')
    parser.add_argument('-b', '--budget', type=int, default=100,
                        help='evaluation budget')
    parser.add_argument('-c', '--cost-ratio', type=float, default=cost_ratios, nargs='*',
                        help='relative cost of a low- vs high-fidelity evaluation')
    parser.add_argument('-f', '--force-rerun', action='store_true',
                        help='Force rerunning this experiment. Deletes previous files')
    parser.add_argument('--shortcut', action='store_true',
                        help="Stop optimization when optimum reached based on function's `x_opt`")
    arguments = parser.parse_args()

    # ensure only valid experiment names are passed on
    valid_experiments = []
    for exp in arguments.experiment:
        if exp in FIDELITY_SELECTORS:
            valid_experiments.append(exp)
        else:
            warn(f"Skipping experiment '{exp}', name not recognised.")
    arguments.experiment = valid_experiments

    main(arguments)
