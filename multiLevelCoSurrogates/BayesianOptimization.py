#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

'''
Surrogates.py: A generic wrapper for various surrogate models such as Kriging and RBF
'''

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import cma
import numpy as np
from scipy.special import erf


class EGO:
    def __init__(self, surrogate, ndims, u_bound, l_bound):
        self.surrogate = surrogate
        self.ndims = ndims
        self.y_best = min(surrogate.y)
        self.x0 = [(u+l)/2 for u, l in zip(u_bound, l_bound)]
        self.sigma0 = 0.9*min([u-l for u, l in zip(u_bound, l_bound)])
        self.cma_opts = {'bounds': [l_bound, u_bound],
                         'maxiter': 250,
                         'verbose': -9,
                         'verb_disp': 0,
                         'verb_log': 0,
                         'verb_time': False}


    def next_infill(self, verbose=False):

        if verbose:
            print('CMA-ES run for optimizing EI...')

        es = cma.CMAEvolutionStrategy(self.x0, self.sigma0, inopts=self.cma_opts).optimize(self.exp_imp)
        x, EI = es.result.xbest, es.result.fbest
        if verbose:
            print('CMA-ES run complete.')
        return x, EI

    def exp_imp(self, x):
        x = x.reshape(1,-1)
        y_hat = self.surrogate.predict(x)
        SSqr = self.surrogate.predict(x, mode='std')

        if SSqr == 0:
            EI = 0
        else:
            EI = (self.y_best - y_hat) * (0.5 + 0.5*erf((self.y_best - y_hat)/np.sqrt(2 * SSqr))) + \
                    np.sqrt(0.5*SSqr/np.pi)*np.exp(-0.5*(self.y_best - y_hat)**2/SSqr)
            EI = -EI

        return EI
