#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
New attempt at using Bayesian optimization using the standard 'bayesian-optimization' package.
"""

import numpy as np
from functools import partial

from multiLevelCoSurrogates.Surrogates import Surrogate, CoSurrogate
from multiLevelCoSurrogates.config import data_dir, folder_name, suffix, data_ext, fit_funcs, fit_func_dims
from multiLevelCoSurrogates.CandidateArchive import CandidateArchive
from multiLevelCoSurrogates.__main__ import createScaledLHS
from multiLevelCoSurrogates.Utils import plotsurfaces


boha = fit_funcs['bohachevsky']
bounds = {'x': (boha.l_bound[0]//20, boha.u_bound[0]//20),
          'y': (boha.l_bound[1]//20, boha.u_bound[1]//20)}

def fit_func(x, y):
    return -boha.high([x, y])


num_init_points = 5
num_iters = 15
num_next_iters = 5


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





bo = BayesianOptimization(fit_func, bounds)

bo.explore({'x': [-1, 3], 'y': [-2, 2]}, eager=True)
bo.maximize(init_points=0, n_iter=0, kappa=2, acq='ei')


# bo.initialize(
#     {
#         'target': [-1, -1],
#         'x': [1, 1],
#         'y': [0, 2]
#     }
# )


count = 0

for i in range(num_init_points):
    point = bo.space.random_points(1)[0]
    bo.explore({'x': [point[0]], 'y': [point[1]]}, eager=True)
    bo.gp.fit(bo.space.X, bo.space.Y)
    plotstuff(fit_func, bo, count)
    count += 1

for i in range(num_iters):
    bo.maximize(init_points=0, n_iter=1, kappa=2, acq='ei')
    plotstuff(fit_func, bo, count)
    count += 1


# The output values can be accessed with self.res
print(bo.res['max'])

# Making changes to the gaussian process can impact the algorithm dramatically.
gp_params = {'kernel': None,
             'alpha': 1e-5}

# Run it again with different acquisition function
for i in range(num_iters):
    bo.maximize(n_iter=1, acq='ei', **gp_params)
    plotstuff(fit_func, bo, count)
    count += 1

# Finally, we take a look at the final results.
print(bo.res['max'])
print(bo.res['all'])
