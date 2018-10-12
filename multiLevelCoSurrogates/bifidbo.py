#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
bifidbo.py: Bi-Fidelity Bayesian Optimization
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import numpy as np
import bayes_opt as bo
from bayes_opt import BayesianOptimization
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process.kernels import Matern
from multiLevelCoSurrogates.Surrogates import Kriging



N_RESTARTS = 100



def blank_fitness(x):
    return None

class BiFidBayesianOptimization:

    def __init__(self, f_low, f_high, cand_arch, acq=None, bounds=None):
        self.f_low = f_low
        self.f_high = f_high
        self.cand_arch = cand_arch
        self.bounds = {'x': (-5,5), 'y':(-5,5)} if bounds is None else bounds

        kind, kappa, xi = 'ucb', 2.576, 0.0

        if acq is not None:
            kind, param = acq
            if kind == 'ucb':
                kappa = param
            elif kind == 'ei':
                xi = param
        self.kind = kind

        self.acq = bo.helpers.UtilityFunction(kind=kind, kappa=kappa, xi=xi).utility
        self.bo_diff = BayesianOptimization(blank_fitness, self.bounds, verbose=False)

        gp_opts = {
            'n': 0,
            'kernel': Matern(nu=2.5),
            'random_state': None,
            'candidate_archive': None,
            'n_restarts_optimizer': N_RESTARTS,
        }

        self.gp_low = Kriging(**gp_opts)
        self.gp_high = Kriging(**gp_opts)
        self.bo_diff.gp = Kriging(**gp_opts)

        candidates, fitnesses = self.cand_arch.getcandidates(n=0, fidelity=['high', 'low'])
        y_high, y_low = fitnesses[:,0], fitnesses[:,1]
        self.rho = self.determine_rho(y_high, y_low)

        diffs = y_high-self.rho*y_low
        self.bo_diff.initialize({
            'x': candidates[:, 0].flatten(),
            'y': candidates[:, 1].flatten(),
            'target': diffs.flatten(),
        })
        self.train_diff()
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

        self.rho = self.determine_rho(y_high, y_low)

        diffs = y_high - self.rho*y_low
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
            'bounds': np.array(list(self.bounds.values()), dtype=np.float),
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

        elif which_model == 'diff':
            kwargs['gp'] = self.bo_diff.gp
            kwargs['y_max'] = self.bo_diff.space.Y.max()


        return bo.helpers.acq_max(**kwargs)
