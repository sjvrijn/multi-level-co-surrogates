#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
New attempt at using Bayesian optimization using the standard 'bayesian-optimization' package.
"""

import numpy as np
from functools import partial
from pyDOE import lhs
from pprint import pprint
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from multiLevelCoSurrogates.config import fit_funcs, fit_func_dims
from multiLevelCoSurrogates.CandidateArchive import CandidateArchive
from multiLevelCoSurrogates.Utils import plotsurfaces


import sys
sys.path.append("./")
from bayes_opt import BayesianOptimization





def plotstuff(fit_func, bo, count):
    funcs = [
        lambda x: fit_func(*x[0]),
        partial(bo.util.utility, gp=bo.gp, y_max=bo.space.Y.max()),
        lambda x: bo.gp.predict(x)[0],
        lambda x: bo.gp.predict(x, return_std=True)[1],
    ]
    titles = [
        f'Function',
        f'ACQ:{bo.util.kind} {count}',
        f'GP {count}',
        f'GP var {count}'
    ]
    plotsurfaces(funcs, titles, (2, 2))


def plotmorestuff(fit_func_high, fit_func_low, bifidbo, count):
    funcs = [
        lambda x: fit_func_high(*x[0]),
        lambda x: fit_func_low(*x[0]),
        lambda x: fit_func_high(*x[0]) - fit_func_low(*x[0]),

        partial(bifidbo.bo_high.util.utility, gp=bifidbo.bo_high.gp, y_max=bifidbo.bo_high.space.Y.max()),
        lambda x: bifidbo.bo_high.gp.predict(x)[0],
        lambda x: bifidbo.bo_high.gp.predict(x, return_std=True)[1],

        partial(bifidbo.bo_low.util.utility, gp=bifidbo.bo_low.gp, y_max=bifidbo.bo_low.space.Y.max()),
        lambda x: bifidbo.bo_low.gp.predict(x)[0],
        lambda x: bifidbo.bo_low.gp.predict(x, return_std=True)[1],

        lambda x: bifidbo.utility(x),
        lambda x: bifidbo.predict_hierarchical(x),
        lambda x: bifidbo.predict_hierarchical(x, return_std=True),
    ]
    titles = [
        f'Function high',
        f'Function low',
        f'Function diff',

        f'High ACQ:{bifidbo.bo_high.util.kind} {count}',
        f'High GP {count}',
        f'High GP var {count}',

        f'Low ACQ:{bifidbo.bo_low.util.kind} {count}',
        f'Low GP {count}',
        f'Low GP var {count}',

        f'Hierarchical ACQ:{bifidbo.bo_diff.util.kind} {count}',
        f'Hierarchical GP {count}',
        f'Hierarchical GP var {count}',
    ]
    plotsurfaces(funcs, titles, (4, 3))

boha = fit_funcs['bohachevsky']
bounds = {'x': (boha.l_bound[0]//20, boha.u_bound[0]//20),
          'y': (boha.l_bound[1]//20, boha.u_bound[1]//20)}


def boexample(num_init_points=5, num_iters=25):

    def fit_func(x, y):
        return -boha.high([x, y])

    bo = BayesianOptimization(fit_func, bounds)
    bo.explore({'x': [-1, 3], 'y': [-2, 2]}, eager=True)
    bo.maximize(init_points=0, n_iter=0, kappa=2)

    for count in range(1, num_init_points+1):
        bo.explore_random(1, eager=True)
        bo.gp.fit(bo.space.X, bo.space.Y)
        plotstuff(fit_func, bo, count)

    for count in range(num_init_points, num_init_points+num_iters+1):
        bo.maximize(init_points=0, n_iter=1, kappa=2)
        plotstuff(fit_func, bo, count)

    # Finally, we take a look at the final results.
    print(bo.res['max'])
    print(bo.res['all'])



# ------------------------------------------------------------------------------

class BiFidBayesianOptimization:

    def __init__(self, bo_low, bo_high, f_low, f_high, cand_arch):
        self.bo_low = bo_low
        self.bo_high = bo_high
        self.f_low = f_low
        self.f_high = f_high
        self.cand_arch = cand_arch

        self.bo_diff = BayesianOptimization(lambda x: None, bounds)

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


    def train_diff(self):
        if self.cand_arch.num_fidelities != 2:
            raise ValueError('Cannot work with anything other than 2 fidelities for now. Sorry :)')

        candidates, fitnesses = self.cand_arch.getcandidates(n=0, fidelity=['high', 'low'])
        y_high, y_low = fitnesses[:,0], fitnesses[:,1]

        self.determine_rho(y_high, y_low)

        diffs = y_high - y_low
        self.bo_diff.gp.fit(candidates, diffs)


    @staticmethod
    def determine_rho(y_high, y_low):

        regr = LinearRegression()
        regr.fit(y_low.reshape(-1, 1), y_high.reshape(-1, 1))
        return regr.coef_.flatten()[0]


    def predict_hierarchical(self, X, return_std=False, y_low=None):
        idx = 1 if return_std else 0

        if y_low is None:
            y_low = self.rho * self.bo_low.gp.predict(X, return_std=return_std)[idx]

        return y_low + self.bo_diff.gp.predict(X, return_std=return_std)[idx]


    def utility(self, X):
        util_low = self.rho * self.bo_low.util.utility(X, gp=self.bo_low.gp, y_max=self.bo_low.space.Y.max())
        util_diff = self.bo_diff.util.utility(X, gp=self.bo_diff.gp, y_max=self.bo_diff.space.Y.max())
        return util_low + util_diff










def bifid_boexample():

    def fit_func_high(x, y):
        return -boha.high([x, y])

    def fit_func_low(x, y):
        return -boha.low([x, y])

    ndim = 2
    num_low_samples = 25
    num_high_samples = 5
    # scaler = MinMaxScaler().fit([[-5,-5], [5,5]])


    bo_low = BayesianOptimization(fit_func_low, bounds)
    bo_high = BayesianOptimization(fit_func_high, bounds)
    archive = CandidateArchive(ndim, fidelities=['high', 'low'])

    low_sample = lhs(ndim, num_low_samples)
    # low_sample = scaler.inverse_transform(low_sample)
    high_sample = low_sample[np.random.choice(low_sample.shape[0], size=num_high_samples, replace=False), :]

    low_out = np.array([[fit_func_low(*x)] for x in low_sample])
    high_out = np.array([[fit_func_high(*x)] for x in high_sample])

    for candidate, result in zip(low_sample, low_out):
        archive.addcandidate(candidate, result, fidelity='low')
    for candidate, result in zip(high_sample, high_out):
        archive.updatecandidate(candidate, result, fidelity='high')

    bifidbo = BiFidBayesianOptimization(bo_low=bo_low, bo_high=bo_high,
                                        f_low=fit_func_low, f_high=fit_func_high,
                                        cand_arch=archive)

    bo_low.initialize({
        'x': low_sample[:,0].flatten(),
        'y': low_sample[:,1].flatten(),
        'target': low_out.flatten(),
    })
    bo_high.initialize({
        'x': high_sample[:, 0].flatten(),
        'y': high_sample[:, 1].flatten(),
        'target': high_out.flatten(),
    })


    bo_low.maximize(0,0)
    bo_high.maximize(0,0)

    plotmorestuff(fit_func_high, fit_func_low, bifidbo, 0)



if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    bifid_boexample()
