#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
New attempt at using Bayesian optimization using the standard 'bayesian-optimization' package.
"""

from itertools import product
from collections import namedtuple
from functools import partial
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pyDOE import lhs
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import Matern

from multiLevelCoSurrogates.config import fit_funcs
from multiLevelCoSurrogates.local import base_dir
from multiLevelCoSurrogates.CandidateArchive import CandidateArchive
from multiLevelCoSurrogates.Surrogates import Kriging
from multiLevelCoSurrogates.Utils import createsurfaces, diffsurface, plotsurfaces, \
    ValueRange, linearscaletransform, select_subsample
from multiLevelCoSurrogates.bifidbo import BiFidBayesianOptimization


N_RESTARTS = 100

def gpplot(x, func, return_std=False):
    idx = 1 if return_std else 0
    return func(x, return_std=return_std)[idx]


ScatterPoints = namedtuple('ScatterPoints', ['x_y', 'z', 'style'])


def plotmorestuff(surfaces, bifidbo, *, count=None, save_as=None, **plot_opts):
    if count is None:
        count = ''

    funcs = [
        *surfaces,

        partial(bifidbo.acq, gp=bifidbo.gp_high, y_max=bifidbo.cand_arch.max['high']),
        partial(gpplot, func=bifidbo.gp_high.predict),
        partial(gpplot, func=bifidbo.gp_high.predict, return_std=True),

        partial(bifidbo.acq, gp=bifidbo.gp_low, y_max=bifidbo.cand_arch.max['low']),
        partial(gpplot, func=bifidbo.gp_low.predict),
        partial(gpplot, func=bifidbo.gp_low.predict, return_std=True),

        partial(bifidbo.acq, gp=bifidbo.bo_diff.gp, y_max=bifidbo.bo_diff.space.Y.max()),
        partial(gpplot, func=bifidbo.bo_diff.gp.predict),
        partial(gpplot, func=bifidbo.bo_diff.gp.predict, return_std=True),

        partial(bifidbo.acq, gp=bifidbo, y_max=bifidbo.bo_diff.space.Y.max()),
        partial(gpplot, func=bifidbo.predict),
        partial(gpplot, func=bifidbo.predict, return_std=True),
    ]
    titles = [
        f'Function high',
        f'Function low',
        f'Function diff',

        f'High ACQ:{bifidbo.kind} {count}',
        f'High GP {count}',
        f'High GP var {count}',

        f'Low ACQ:{bifidbo.kind} {count}',
        f'Low GP {count}',
        f'Low GP var {count}',

        f'Diff ACQ:{bifidbo.kind} {count}',
        f'Diff GP {count}',
        f'Diff GP var {count}',

        f'Hierarchical ACQ:{bifidbo.kind} {count}',
        f'Hierarchical GP {count}',
        f'Hierarchical GP var {count}',
    ]
    surfaces = createsurfaces(funcs)

    p_high = [ScatterPoints(*bifidbo.cand_arch.getcandidates(fidelity='high'),
                            style={'marker': 'o', 'facecolors': 'none', 'color': 'red'})]
    p_low = [ScatterPoints(*bifidbo.cand_arch.getcandidates(fidelity='low'),
                           style={'marker': '+', 'color': 'red'})]
    p_both = [
        p_high[0],
        p_low[0],
    ]
    points = [
        p_high, p_low, p_both,
        p_high, p_high, p_high,
        p_low, p_low, p_low,
        p_both, p_both, p_both,
        p_both, p_both, p_both,
    ]

    if save_as:
        savename_2d = f'{base_dir}{save_as}_2d_{count}.png'
        savename_3d = f'{base_dir}{save_as}_3d_{count}.png'
    else:
        savename_2d = savename_3d = None

    plot_shape = (5, 3)

    if plot_opts.get('plot_2d', False):
        plotsurfaces(surfaces, all_points=points,
                     titles=titles, shape=plot_shape,
                     save_as=savename_2d, as_3d=False, **plot_opts)
    if plot_opts.get('plot_3d', False):
        plotsurfaces(surfaces, all_points=points,
                     titles=titles, shape=plot_shape,
                     save_as=savename_3d, as_3d=True, **plot_opts)


boha = fit_funcs['himmelblau']
bounds = {
    'x': (boha.l_bound[0], boha.u_bound[0]),
    'y': (boha.l_bound[1], boha.u_bound[1]),
}


def row_vectorize(func):
    def new_func(X):
        try:
            return np.array([func(row) for row in X])
        except TypeError:
            return func(X)
    return new_func


def fit_func_high(x):
    high = row_vectorize(boha.high)
    return -high(x)

def fit_func_low(x):
    low = row_vectorize(boha.low)
    return -low(x)


funcs = [
    fit_func_high,
    fit_func_low,
]
surfaces = createsurfaces(funcs)
surfaces.append(diffsurface(surfaces[0], surfaces[1]))


def createbifidbo(num_low_samples=25, num_high_samples=5, plot_surfaces=False, acq=None):

    if num_high_samples < 3:
        raise ValueError('At least 3 high-fidelity datapoints are needed to avoid numerical instability')
    if num_low_samples <= num_high_samples:
        raise ValueError('Please provide more low-fidelity than high-fidelity samples')

    ndim = 2
    range_in = ValueRange(-5, 5)
    range_lhs = ValueRange(0, 1)

    gp_low = Kriging(candidate_archive=None, n=0, kernel=Matern(nu=2.5),
                     n_restarts_optimizer=N_RESTARTS, random_state=None)
    gp_high = Kriging(candidate_archive=None, n=0, kernel=Matern(nu=2.5),
                      n_restarts_optimizer=N_RESTARTS, random_state=None)
    archive = CandidateArchive(ndim, fidelities=['high', 'low'])

    low_sample = lhs(ndim, num_low_samples)
    low_sample = linearscaletransform(low_sample, range_in=range_lhs, range_out=range_in)

    high_sample = select_subsample(low_sample.T, num_high_samples).T

    low_out = fit_func_low(low_sample)
    high_out = fit_func_high(high_sample)

    for candidate, result in zip(low_sample, low_out):
        archive.addcandidate(candidate, result, fidelity='low')
    for candidate, result in zip(high_sample, high_out):
        archive.addcandidate(candidate, result, fidelity='high')

    bifidbo = BiFidBayesianOptimization(
        gp_low=gp_low, gp_high=gp_high,
        f_low=fit_func_low, f_high=fit_func_high,
        cand_arch=archive, acq=acq, bounds=bounds
    )

    # Fit GP
    bifidbo.train_gp(fidelity='low')
    bifidbo.train_gp(fidelity='high')

    if plot_surfaces:
        plotmorestuff(surfaces, bifidbo, count=0)
    return bifidbo


def find_infill_and_retrain(bifidbo, which_model='hierarchical', fidelity='low'):
    infill_in = bifidbo.acq_max(which_model=which_model)

    if fidelity == 'low':
        infill_out = fit_func_low(infill_in)
    elif fidelity == 'high':
        infill_out = fit_func_high(infill_in)
    elif fidelity == 'both':
        infill_out = [fit_func_low(infill_in), fit_func_high(infill_in)]
        fidelity = ['low', 'high', 'diff']
    else:
        raise ValueError(f"fidelity '{fidelity}' not recognized")

    bifidbo.cand_arch.addcandidate(candidate=infill_in, fitness=infill_out, fidelity=fidelity)
    bifidbo.train_gp(fidelity=fidelity)


def calc_mse(bifidbo, test_mse_high, test_mse_low, test_sample, verbosity=0):
    if verbosity > 0:
        print('        Predicting...')
    low_predict_values = bifidbo.gp_low.predict(test_sample)
    high_predict_values = bifidbo.gp_high.predict(test_sample)
    hierarchical_predict_values = bifidbo.predict(test_sample)
    if verbosity > 0:
        print('        Calculating MSE...')
    mse_low = test_mse_low(low_predict_values)
    mse_high = test_mse_high(high_predict_values)
    mse_hierarchical = test_mse_high(hierarchical_predict_values)
    mse_low_on_high = test_mse_high(low_predict_values)
    return mse_hierarchical, mse_high, mse_low, mse_low_on_high



MSERecord = namedtuple('MSERecord', ['which_model', 'fidelity', 'repetition', 'iteration',
                                     'mse_high', 'mse_low', 'mse_hier', 'mse_low_on_high'])

def infill_experiment(num_repetitions=10, num_iters=1, which_model='hierarchical', fidelity='low', acq=None,
                      *, verbosity=0, **plot_opts):

    if verbosity > 0:
        print(f'--------------------------------------------------------------------------------\n'
              f'Updating {fidelity} for {num_iters} steps, based on {which_model}, repeated {num_repetitions} times.\n'
              f'---')

    make_plots = plot_opts.get('plot_2d', False) or plot_opts.get('plot_3d', False)

    range_in = ValueRange(-5, 5)
    range_lhs = ValueRange(0, 1)

    # TODO: is 1k samples good/enough?
    test_sample = linearscaletransform(lhs(n=2, samples=1000), range_in=range_lhs, range_out=range_in)

    test_mse_high = partial(mean_squared_error, y_pred=fit_func_high(test_sample))
    test_mse_low = partial(mean_squared_error, y_pred=fit_func_low(test_sample))

    records = []

    if 'both' in fidelity:
        interval = max(int(fidelity[4:]), 1)
    else:
        interval = None

    import progressbar
    save_as = f"{acq[0]}{acq[1] if acq[1] is not None else ''}_{fidelity}_{which_model}"

    with progressbar.ProgressBar(max_value=num_repetitions*(num_iters + 1)) as bar:
        for rep in range(num_repetitions):

            bar.update(rep*num_repetitions + 0)
            if verbosity > 1:
                print(f'Repetition {rep}/{num_repetitions}:')
                print('    Creating Bi-Fid BO')
            bifidbo = createbifidbo(num_low_samples=5, num_high_samples=3, acq=acq)

            mse_hierarchical, mse_high, mse_low, mse_low_on_high = calc_mse(bifidbo, test_mse_high, test_mse_low, test_sample, verbosity=verbosity-2)
            records.append(MSERecord(which_model, fidelity, rep, iteration=0,
                                     mse_low=mse_low, mse_high=mse_high, mse_hier=mse_hierarchical, mse_low_on_high=mse_low_on_high))
            if make_plots:
                plotmorestuff(surfaces, bifidbo, count=0, save_as=save_as+f'_r{rep}', **plot_opts)

            for i in range(1, num_iters + 1):
                bar.update(rep*num_repetitions + i)

                if verbosity > 2:
                    print(f'    Iteration {i}/{num_iters}')
                    print('        Finding infill...')

                if interval is None:
                    fid = fidelity
                elif i % interval == 0:
                    fid = 'both'
                else:
                    fid = 'low'

                find_infill_and_retrain(bifidbo, which_model=which_model, fidelity=fid)

                mse_hierarchical, mse_high, mse_low, mse_low_on_high = calc_mse(bifidbo, test_mse_high, test_mse_low, test_sample, verbosity=verbosity-3)
                records.append(MSERecord(which_model, fidelity, rep, iteration=i,
                                         mse_low=mse_low, mse_high=mse_high, mse_hier=mse_hierarchical, mse_low_on_high=mse_low_on_high))
                if make_plots:
                    plotmorestuff(surfaces, bifidbo, count=i, save_as=save_as+f'_r{rep}', **plot_opts)

            if verbosity > 1:
                print()

    return records


def flatten(iterable):
    return [x for y in iterable for x in y]


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    run_opts = {
        'num_repetitions': 15,
        'num_iters': 100,
        'verbosity': 1,
    }
    plot_opts = {
        'plot_2d': False,
        'plot_3d': False,
        'show': False,
    }

    acqs = [
        # ('ucb', 0.5),
        # ('ucb', 1.0),
        ('ucb', 1.5),
        # ('ucb', 2.0),
        # ('ucb', 3.0),
        # ('ucb', 4.0),
        # ('ucb', 5.0),
        # ('ei', 0.0),
        ('ei', 0.5),
        # ('ei', 1.0),
        # ('poi', None),
    ]

    fids_and_models = [
        ('high', 'high'),
        # ('both 1', 'hierarchical'),
        ('both 3', 'hierarchical'),
        # ('both 1', 'diff'),
        ('both 3', 'diff'),
    ]

    print(acqs)
    records = []
    for acq, fid_and_model in product(acqs, fids_and_models):
        fid, model = fid_and_model
        run_opts['acq'] = acq
        print(acq, fid, model)
        # TODO: make initial sample depend on an optional random seed
        records.append(infill_experiment(fidelity=fid, which_model=model, **run_opts, **plot_opts))

    df = pd.DataFrame(flatten(records))
    df.to_csv(base_dir+f"_records.csv", index_label='index')
    # df.to_csv(base_dir+f"{name}{param if param is not None else ''}_diff_records.csv", index_label='index')
