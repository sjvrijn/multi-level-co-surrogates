#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
New attempt at using Bayesian optimization using the standard 'bayesian-optimization' package.
"""

import numpy as np
from functools import partial
from sklearn.linear_model import LinearRegression

from multiLevelCoSurrogates.config import fit_funcs, fit_func_dims
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

    def __init__(self, bayes_low, bayes_high, f_low, f_high, cand_arch):
        self.bayes_low = bayes_low
        self.bayes_high = bayes_high
        self.f_low = f_low
        self.f_high = f_high
        self.cand_arch = cand_arch

        self.bayes_diff = BayesianOptimization(lambda x: None, bounds)

        candidates, fitnesses = self.cand_arch.getcandidates(n=0, fidelity=['high', 'low'])
        y_high, y_low = fitnesses[:,0], fitnesses[:,1]
        self.rho = self.determine_rho(y_high, y_low)


    def train_diff(self):
        if self.cand_arch.num_fidelities != 2:
            raise ValueError('Cannot work with anything other than 2 fidelities for now. Sorry :)')

        candidates, fitnesses = self.cand_arch.getcandidates(n=0, fidelity=['high', 'low'])
        y_high, y_low = fitnesses[:,0], fitnesses[:,1]

        self.determine_rho(y_high, y_low)

        diffs = y_high - y_low
        self.bayes_diff.gp.fit(candidates, diffs)


    @staticmethod
    def determine_rho(y_high, y_low):

        regr = LinearRegression()
        regr.fit(y_low.reshape(-1, 1), y_high.reshape(-1, 1))
        return regr.coef_.flatten()[0]


    def predict_hierarchical(self, X, return_std=False, y_low=None):
        if y_low is None:
            y_low = self.rho * self.bayes_low.predict(X, return_std=return_std)

        return y_low + self.bayes_diff.gp.predict(X, return_std=return_std)












def bifid_boexample(num_init_points=5, num_iters=25):

    def fit_func_high(x, y):
        return -boha.high([x, y])

    def fit_func_low(x, y):
        return -boha.high([x, y])

    bo_low = BayesianOptimization(fit_func_low, bounds)
    bo_low.explore({'x': [-1, 3], 'y': [-2, 2]}, eager=True)
    bo_low.maximize(init_points=0, n_iter=0, kappa=2)

    for count in range(1, num_init_points+1):
        bo_low.explore_random(1, eager=True)
        bo_low.gp.fit(bo_low.space.X, bo_low.space.Y)
        plotstuff(fit_func_low, bo_low, count)

    for count in range(num_init_points, num_init_points+num_iters+1):
        bo_low.maximize(init_points=0, n_iter=1, kappa=2)
        plotstuff(fit_func_low, bo_low, count)

    # Finally, we take a look at the final results.
    print(bo_low.res['max'])
    print(bo_low.res['all'])





if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    boexample()
