#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
New attempt at using Bayesian optimization using the standard 'bayesian-optimization' package.
"""

import numpy as np
from functools import partial
from pyDOE import lhs
from pprint import pprint
from sklearn.utils import check_random_state
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from multiLevelCoSurrogates.config import fit_funcs, fit_func_dims
from multiLevelCoSurrogates.local import base_dir
from multiLevelCoSurrogates.CandidateArchive import CandidateArchive
from multiLevelCoSurrogates.Utils import createsurface, diffsurface, plotsurfaces, \
    ValueRange, linearscaletransform, select_subsample


import sys
sys.path.append("./")
import bayes_opt as bo
from bayes_opt import BayesianOptimization





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
    plotsurfaces(funcs, titles, (2, 2))


def gpplot(x, func, return_std=False):
    idx = 1 if return_std else 0
    return func(x, return_std=return_std)[idx]


def plotmorestuff(surfaces, bifidbo, count):
    funcs = [
        *surfaces,

        partial(bifidbo.acq.utility, gp=bifidbo.gp_high, y_max=bifidbo.cand_arch.max['high']),
        partial(gpplot, func=bifidbo.gp_high.predict),
        partial(gpplot, func=bifidbo.gp_high.predict, return_std=True),

        partial(bifidbo.acq.utility, gp=bifidbo.gp_low, y_max=bifidbo.cand_arch.max['low']),
        partial(gpplot, func=bifidbo.gp_low.predict),
        partial(gpplot, func=bifidbo.gp_low.predict, return_std=True),

        partial(bifidbo.acq.utility, gp=bifidbo.bo_diff.gp, y_max=bifidbo.bo_diff.space.Y.max()),
        partial(gpplot, func=bifidbo.bo_diff.gp.predict),
        partial(gpplot, func=bifidbo.bo_diff.gp.predict, return_std=True),

        bifidbo.utility,
        bifidbo.predict,
        partial(bifidbo.predict, return_std=True),
    ]
    titles = [
        f'Function high',
        f'Function low',
        f'Function diff',

        f'High ACQ:{bifidbo.acq.kind} {count}',
        f'High GP {count}',
        f'High GP var {count}',

        f'Low ACQ:{bifidbo.acq.kind} {count}',
        f'Low GP {count}',
        f'Low GP var {count}',

        f'Diff ACQ:{bifidbo.acq.kind} {count}',
        f'Diff GP {count}',
        f'Diff GP var {count}',

        f'Hierarchical ACQ:{bifidbo.acq.kind} {count}',
        f'Hierarchical GP {count}',
        f'Hierarchical GP var {count}',
    ]

    # plotsurfaces(funcs, titles, (5, 3), save_as=f'{base_dir}plotmorestuff_2d_{count}.png', as_3d=False)
    # plotsurfaces(funcs, titles, (5, 3), save_as=f'{base_dir}plotmorestuff_3d_{count}.png', as_3d=True)

    plotsurfaces(funcs, titles, (5, 3), as_3d=False)
    # plotsurfaces(funcs, titles, (5, 3), as_3d=True)

boha = fit_funcs['bohachevsky']
bounds = {'x': (boha.l_bound[0]//20, boha.u_bound[0]//20),
          'y': (boha.l_bound[1]//20, boha.u_bound[1]//20)}


def boexample(num_init_points=5, num_iters=25):

    def fit_func(x, y):
        return -boha.high([x, y])

    bopt = BayesianOptimization(fit_func, bounds)
    bopt.explore({'x': [-1, 3], 'y': [-2, 2]}, eager=True)
    bopt.maximize(init_points=0, n_iter=0, kappa=2)

    for count in range(1, num_init_points+1):
        bopt.explore_random(1, eager=True)
        bopt.gp.fit(bopt.space.X, bopt.space.Y)
        plotstuff(fit_func, bopt, count)

    for count in range(num_init_points, num_init_points+num_iters+1):
        bopt.maximize(init_points=0, n_iter=1, kappa=2)
        plotstuff(fit_func, bopt, count)

    # Finally, we take a look at the final results.
    print(bopt.res['max'])
    print(bopt.res['all'])



# ------------------------------------------------------------------------------

def emptyfit(x):
    return None

class BiFidBayesianOptimization:

    def __init__(self, gp_low, gp_high, f_low, f_high, cand_arch):
        self.gp_low = gp_low
        self.gp_high = gp_high
        self.f_low = f_low
        self.f_high = f_high
        self.cand_arch = cand_arch

        self.acq = bo.helpers.UtilityFunction(kind='ucb', kappa=2.576, xi=0.0)

        self.bo_diff = BayesianOptimization(emptyfit, bounds)

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
        idx = 1 if return_std else 0

        if y_low is None:
            y_low = self.rho * self.gp_low.predict(X, return_std=return_std)[idx]

        return y_low + self.bo_diff.gp.predict(X, return_std=return_std)[idx]


    def utility(self, X, gp=None, y_max=None):
        util_low = self.rho * self.acq.utility(X, gp=self.gp_low, y_max=self.cand_arch.max['low'])
        util_diff = self.acq.utility(X, gp=self.bo_diff.gp, y_max=self.bo_diff.space.Y.max())
        return util_low + util_diff


    def acq_max(self):
        return bo.helpers.acq_max(ac=self.utility,
                                  gp=self,
                                  y_max=self.cand_arch.max['high'],
                                  bounds=np.array(list(bounds.values()), dtype=np.float),
                                  random_state=self.bo_diff.random_state)








def fit_func_high(x, y):
    return -boha.high([x, y])

def fit_func_low(x, y):
    return -boha.low([x, y])

funcs = [
    lambda x: fit_func_high(*x[0]),
    lambda x: fit_func_low(*x[0]),
]
surfaces = list(map(createsurface, funcs))
surfaces.append(diffsurface(surfaces[0], surfaces[1]))


def createbifidbo():

    ndim = 2
    num_low_samples = 25
    num_high_samples = 5
    range_in = ValueRange(-5, 5)
    range_lhs = ValueRange(0, 1)


    gp_low = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=25, random_state=None)
    gp_high = GaussianProcessRegressor(kernel=Matern(nu=2.5), n_restarts_optimizer=25, random_state=None)
    archive = CandidateArchive(ndim, fidelities=['high', 'low'])

    low_sample = lhs(ndim, num_low_samples)
    low_sample = linearscaletransform(low_sample, range_in=range_lhs, range_out=range_in)

    high_sample = select_subsample(low_sample.T, num_high_samples).T

    low_out = np.array([[fit_func_low(*x)] for x in low_sample])
    high_out = np.array([[fit_func_high(*x)] for x in high_sample])

    for candidate, result in zip(low_sample, low_out):
        archive.addcandidate(candidate, result, fidelity='low')
    for candidate, result in zip(high_sample, high_out):
        archive.updatecandidate(candidate, result, fidelity='high')

    bifidbo = BiFidBayesianOptimization(gp_low=gp_low, gp_high=gp_high,
                                        f_low=fit_func_low, f_high=fit_func_high,
                                        cand_arch=archive)

    # Fit GP
    bifidbo.train_gp(fidelity='low')
    bifidbo.train_gp(fidelity='high')

    plotmorestuff(surfaces, bifidbo, 0)
    return bifidbo



def optimize(bifidbo, surfs, num_steps=10):

    for count in range(1, num_steps+1):
        argmax = bifidbo.acq_max()
        bifidbo.cand_arch.addcandidate(argmax, fit_func_low(*argmax), fidelity='low')

        bifidbo.train_gp(fidelity='low')

        plotmorestuff(surfs, bifidbo, count)


def find_infill_and_retrain(bifidbo):
    pass


def infill_experiment(num_repetitions=10):
    range_in = ValueRange(-5, 5)
    range_lhs = ValueRange(0, 1)
    test_sample = lhs(n=2, samples=250)
    test_sample = linearscaletransform(test_sample, range_in=range_lhs, range_out=range_in)

    test_values = [fit_func_high(*sample) for sample in test_sample]

    for rep in range(num_repetitions):
        bifidbo = createbifidbo()
        low_predict_values = bifidbo.gp_low.predict(test_sample)
        high_predict_values = bifidbo.gp_high.predict(test_sample)
        hierarchical_predict_values = bifidbo.predict(test_sample)

        pre_mse_low = mean_squared_error(test_values, low_predict_values)
        pre_mse_high = mean_squared_error(test_values, high_predict_values)
        pre_mse_hierarchical = mean_squared_error(test_values, hierarchical_predict_values)


        find_infill_and_retrain(bifidbo)


        low_predict_values = bifidbo.gp_low.predict(test_sample)
        high_predict_values = bifidbo.gp_high.predict(test_sample)
        hierarchical_predict_values = bifidbo.predict(test_sample)

        post_mse_low = mean_squared_error(test_values, low_predict_values)
        post_mse_high = mean_squared_error(test_values, high_predict_values)
        post_mse_hierarchical = mean_squared_error(test_values, hierarchical_predict_values)

    return ((pre_mse_low, pre_mse_high, pre_mse_hierarchical),
            (post_mse_low, post_mse_high, post_mse_hierarchical))


if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    bifidbo = createbifidbo()
    optimize(bifidbo, surfaces)
