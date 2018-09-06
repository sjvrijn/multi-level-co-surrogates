#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
New attempt at using Bayesian optimization using the standard 'bayesian-optimization' package.
"""

import sys
sys.path.append("./")
sys.path.append("../")
import bayes_opt as bo
from bayes_opt import BayesianOptimization



import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from collections import namedtuple
from functools import partial
from pyDOE import lhs
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import Matern

from multiLevelCoSurrogates.config import fit_funcs
from multiLevelCoSurrogates.local import base_dir
from multiLevelCoSurrogates.CandidateArchive import CandidateArchive
from multiLevelCoSurrogates.Surrogates import Kriging
from multiLevelCoSurrogates.Utils import createsurfaces, diffsurface, plotsurfaces, \
    ValueRange, linearscaletransform, select_subsample






def plotstuff(fit_func, bopt, count):
    funcs = [
        lambda x: fit_func(*x[0]),
        partial(bopt.util.utility, gp=bopt.gp, y_max=bopt.space.Y.max()),
        lambda x: bopt.gp.predict(x)[0],
        lambda x: bopt.gp.predict(x, return_std=True)[1],
    ]
    titles = [
        f'Function',
        f'ACQ:{bopt.util.kind} {count}',
        f'GP {count}',
        f'GP var {count}'
    ]
    surfaces = createsurfaces(funcs)
    plotsurfaces(surfaces, titles, (2, 2))


def gpplot(x, func, return_std=False):
    idx = 1 if return_std else 0
    return func(x, return_std=return_std)[idx]


def plotmorestuff(surfaces, bifidbo, *, count='', save_as=None, plot_2d=True, plot_3d=False):
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

    if save_as:
        savename_2d = f'{base_dir}{save_as}_2d_{count}.png'
        savename_3d = f'{base_dir}{save_as}_3d_{count}.png'
    else:
        savename_2d = savename_3d = None

    if plot_2d:
        plotsurfaces(surfaces, titles, (5, 3), save_as=savename_2d, as_3d=False)
    if plot_3d:
        plotsurfaces(surfaces, titles, (5, 3), save_as=savename_3d, as_3d=True)


boha = fit_funcs['himmelblau']
bounds = {'x': (boha.l_bound[0], boha.u_bound[0]),
          'y': (boha.l_bound[1], boha.u_bound[1])}



# ------------------------------------------------------------------------------

def emptyfit(x):
    return None

class BiFidBayesianOptimization:

    def __init__(self, gp_low, gp_high, f_low, f_high, cand_arch, acq=None):
        self.gp_low = gp_low
        self.gp_high = gp_high
        self.f_low = f_low
        self.f_high = f_high
        self.cand_arch = cand_arch

        kind, kappa, xi = 'ucb', 2.576, 0.0

        if acq is not None:
            kind, param = acq
            if kind == 'ucb':
                kappa = param
            elif kind == 'ei':
                xi = param
        self.kind = kind

        self.acq = bo.helpers.UtilityFunction(kind=kind, kappa=kappa, xi=xi).utility
        self.bo_diff = BayesianOptimization(emptyfit, bounds, verbose=False)
        self.bo_diff.gp = Kriging(candidate_archive=None, n=0, kernel=Matern(nu=2.5), n_restarts_optimizer=25, random_state=None)

        candidates, fitnesses = self.cand_arch.getcandidates(n=0, fidelity=['high', 'low'])
        y_high, y_low = fitnesses[:,0], fitnesses[:,1]
        self.rho = self.determine_rho(y_high, y_low)
        self.train_diff()

        diffs = y_high-y_low
        self.bo_diff.initialize({
            'x': candidates[:, 0].flatten(),
            'y': candidates[:, 1].flatten(),
            'target': diffs.flatten(),
        })
        self.bo_diff.maximize(0,0)


    def train_gp(self, fidelity, n=None):

        if fidelity == 'low':
            gp = self.gp_low
        elif fidelity == 'high':
            gp = self.gp_high
        elif fidelity == 'diff':
            return self.train_diff(n=n)
        elif isinstance(fidelity, list):
            for fid in fidelity:
                self.train_gp(fid, n=n)
            return
        else:
            raise ValueError(f"Fidelity '{fidelity}' unknown, please choose 'high' or 'low'.")

        a, b = self.cand_arch.getcandidates(n=n, fidelity=fidelity)
        gp.fit(a, b.ravel())


    def train_diff(self, n=None):
        if self.cand_arch.num_fidelities != 2:
            raise ValueError('Cannot work with anything other than 2 fidelities for now. Sorry :)')

        candidates, fitnesses = self.cand_arch.getcandidates(n=n, fidelity=['high', 'low'])
        y_high, y_low = fitnesses[:,0], fitnesses[:,1]

        self.determine_rho(y_high, y_low)

        diffs = y_high - y_low
        self.bo_diff.gp.fit(candidates, diffs)


    @staticmethod
    def determine_rho(y_high, y_low):

        regr = LinearRegression()
        regr.fit(y_low.reshape(-1, 1), y_high.reshape(-1, 1))
        return regr.coef_.flatten()[0]


    def predict(self, X, return_std=False, y_low=None):
        if y_low is None:
            low_pred = self.gp_low.predict(X, return_std=return_std)
            if return_std:
                low_std = low_pred[1]
                low_pred = low_pred[0]
            y_low = self.rho * low_pred

        diff_pred = self.bo_diff.gp.predict(X, return_std=return_std)
        if return_std:
            diff_std = diff_pred[1]
            diff_pred = diff_pred[0]
            result = y_low + diff_pred, np.sqrt(low_std**2 + diff_std**2)
        else:
            result = y_low + diff_pred
        return result


    def acq_max(self, which_model):
        kwargs = {
            'ac': self.acq,
            'bounds': np.array(list(bounds.values()), dtype=np.float),
            'random_state': self.bo_diff.random_state,
            'n_warmup': 1000,
            'n_iter': 50
        }

        if which_model == 'hierarchical':
            kwargs['gp'] = self
            kwargs['y_max'] = self.cand_arch.max['high']

        elif which_model == 'low':
            kwargs['gp'] = self.gp_low
            kwargs['y_max'] = self.cand_arch.max['low']

        elif which_model == 'high':
            kwargs['gp'] = self.gp_high
            kwargs['y_max'] = self.cand_arch.max['high']

        return bo.helpers.acq_max(**kwargs)




def fit_func_high(x, y):
    return -boha.high([x, y])

def fit_func_low(x, y):
    return -boha.low([x, y])

funcs = [
    lambda x: fit_func_high(*x[0]),
    lambda x: fit_func_low(*x[0]),
]
surfaces = createsurfaces(funcs)
surfaces.append(diffsurface(surfaces[0], surfaces[1]))


def createbifidbo(num_low_samples=25, num_high_samples=5, plot_surfaces=False, acq=None):

    ndim = 2
    range_in = ValueRange(-5, 5)
    range_lhs = ValueRange(0, 1)

    gp_low = Kriging(candidate_archive=None, n=0, kernel=Matern(nu=2.5), n_restarts_optimizer=25, random_state=None)
    gp_high = Kriging(candidate_archive=None, n=0, kernel=Matern(nu=2.5), n_restarts_optimizer=25, random_state=None)
    archive = CandidateArchive(ndim, fidelities=['high', 'low'])

    low_sample = lhs(ndim, num_low_samples)
    low_sample = linearscaletransform(low_sample, range_in=range_lhs, range_out=range_in)

    high_sample = select_subsample(low_sample.T, num_high_samples).T

    low_out = np.array([[fit_func_low(*x)] for x in low_sample])
    high_out = np.array([[fit_func_high(*x)] for x in high_sample])

    for candidate, result in zip(low_sample, low_out):
        archive.addcandidate(candidate, result, fidelity='low')
    for candidate, result in zip(high_sample, high_out):
        archive.addcandidate(candidate, result, fidelity='high')

    bifidbo = BiFidBayesianOptimization(gp_low=gp_low, gp_high=gp_high,
                                        f_low=fit_func_low, f_high=fit_func_high,
                                        cand_arch=archive, acq=acq)

    # Fit GP
    bifidbo.train_gp(fidelity='low')
    bifidbo.train_gp(fidelity='high')

    if plot_surfaces:
        plotmorestuff(surfaces, bifidbo, count=0)
    return bifidbo



def optimize(bifidbo, surfs, num_steps=10):

    for count in range(1, num_steps+1):
        argmax = bifidbo.acq_max()
        bifidbo.cand_arch.addcandidate(argmax, fit_func_low(*argmax), fidelity='low')

        bifidbo.train_gp(fidelity='low')

        plotmorestuff(surfs, bifidbo, count=count)


def find_infill_and_retrain(bifidbo, which_model='hierarchical', fidelity='low'):
    infill_in = bifidbo.acq_max(which_model=which_model)

    if fidelity == 'low':
        infill_out = fit_func_low(*infill_in)
    elif fidelity == 'high':
        infill_out = fit_func_high(*infill_in)
    elif fidelity == 'both':
        infill_out = [fit_func_low(*infill_in), fit_func_high(*infill_in)]
        fidelity = ['low', 'high', 'diff']
    else:
        raise ValueError(f"fidelity '{fidelity}' not recognized")

    bifidbo.cand_arch.addcandidate(candidate=infill_in, fitness=infill_out, fidelity=fidelity)
    bifidbo.train_gp(fidelity=fidelity)


def calc_mse(bifidbo, test_mse, test_sample, verbosity=0):
    if verbosity > 0:
        print('        Predicting...')
    low_predict_values = bifidbo.gp_low.predict(test_sample)
    high_predict_values = bifidbo.gp_high.predict(test_sample)
    hierarchical_predict_values = bifidbo.predict(test_sample)
    if verbosity > 0:
        print('        Calculating MSE...')
    mse_low = test_mse(low_predict_values)
    mse_high = test_mse(high_predict_values)
    mse_hierarchical = test_mse(hierarchical_predict_values)
    return mse_hierarchical, mse_high, mse_low



MSERecord = namedtuple('MSERecord', ['which_model', 'fidelity', 'repetition', 'iteration',
                                     'mse_high', 'mse_low', 'mse_hier'])

def infill_experiment(num_repetitions=10, num_iterations=1, which_model='hierarchical', fidelity='low', acq=None,
                      *, verbosity=0, make_plots=False):

    if verbosity > 0:
        print(f'--------------------------------------------------------------------------------\n'
              f'Updating {fidelity} for {num_iterations} steps, based on {which_model}, repeated {num_repetitions} times.\n'
              f'---')

    range_in = ValueRange(-5, 5)
    range_lhs = ValueRange(0, 1)
    test_sample = lhs(n=2, samples=250)
    test_sample = linearscaletransform(test_sample, range_in=range_lhs, range_out=range_in)

    test_values = np.array([fit_func_high(*sample) for sample in test_sample])
    test_mse = partial(mean_squared_error, y_pred=test_values)

    records = []

    if 'both' in fidelity:
        interval = max(int(fidelity[4:]), 1)
    else:
        interval = None

    import progressbar
    save_as = f"{acq[0]}{acq[1] if acq[1] is not None else ''}_{fidelity}_{which_model}"
    plot_2d = True
    plot_3d = False

    with progressbar.ProgressBar(max_value=num_repetitions*(num_iterations+1)) as bar:
        for rep in range(num_repetitions):

            bar.update(rep*num_repetitions + 0)
            if verbosity > 1:
                print(f'Repetition {rep}/{num_repetitions}:')
                print('    Creating Bi-Fid BO')
            bifidbo = createbifidbo(num_low_samples=5, num_high_samples=2, acq=acq)

            mse_hierarchical, mse_high, mse_low = calc_mse(bifidbo, test_mse, test_sample, verbosity=verbosity-2)
            records.append(MSERecord(which_model, fidelity, rep, iteration=0,
                                     mse_low=mse_low, mse_high=mse_high, mse_hier=mse_hierarchical))
            if make_plots:
                plotmorestuff(surfaces, bifidbo, count=0, save_as=save_as, plot_2d=plot_2d, plot_3d=plot_3d)

            for i in range(1, num_iterations+1):
                bar.update(rep*num_repetitions + i)

                if verbosity > 2:
                    print(f'    Iteration {i}/{num_iterations}')
                    print('        Finding infill...')

                if interval is None:
                    fid = fidelity
                else:
                    if i % interval == 0:
                        fid = 'both'
                    else:
                        fid = 'low'

                find_infill_and_retrain(bifidbo, which_model=which_model, fidelity=fid)

                mse_hierarchical, mse_high, mse_low = calc_mse(bifidbo, test_mse, test_sample, verbosity=verbosity-3)
                records.append(MSERecord(which_model, fidelity, rep, iteration=i,
                                         mse_low=mse_low, mse_high=mse_high, mse_hier=mse_hierarchical))
                if make_plots:
                    plotmorestuff(surfaces, bifidbo, count=i, save_as=save_as, plot_2d=plot_2d, plot_3d=plot_3d)

            if verbosity > 1:
                print()

    return records


def flatten(iterable):
    return [x for y in iterable for x in y]


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)

    run_opts = {
        'num_repetitions': 3,
        'num_iterations': 25,
        'verbosity': 1,
        'make_plots': False,
    }

    acqs = [
        ('ucb', 0.5),
        ('ucb', 1.0),
        ('ucb', 1.5),
        ('ucb', 2.0),
        ('ucb', 3.0),
        ('ucb', 4.0),
        ('ucb', 5.0),
        ('ei', 0.0),
        ('ei', 0.5),
        ('ei', 1.0),
        # ('poi', None),
    ]

    print(acqs)
    for acq in acqs:
        run_opts['acq'] = acq
        print(acq)
        records = [
            # TODO: make initial sample depend on an optional random seed
            infill_experiment(fidelity='high', which_model='high', **run_opts),
            infill_experiment(fidelity='low', which_model='hierarchical', **run_opts),
            infill_experiment(fidelity='both 1', which_model='hierarchical', **run_opts),
            infill_experiment(fidelity='both 3', which_model='hierarchical', **run_opts),
            infill_experiment(fidelity='both 5', which_model='hierarchical', **run_opts),
        ]

        df = pd.DataFrame(flatten(records))
        name, param = acq
        df.to_csv(base_dir+f"{name}{param if param is not None else ''}_records.csv", index_label='index')
