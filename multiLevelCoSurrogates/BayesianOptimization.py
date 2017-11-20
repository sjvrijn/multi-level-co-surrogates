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
    def __init__(self, surrogate, ndims):
        self.surrogate = surrogate
        self.ndims = ndims
        self.y_best = min(surrogate.y)

    def next_infill(self, verbose=False):

        if verbose:
            print('CMA-ES run for optimizing EI...')
        x0 = [0.5] * self.ndims
        sigma0 = 1
        es = cma.CMAEvolutionStrategy(x0, sigma0).optimize(self.exp_imp)
        x, EI = es.result.xbest, es.result.fbest
        if verbose:
            print('CMA-ES run complete.')
        return x, EI

    def exp_imp(self, x):
        y_hat, SSqr = self.surrogate.predict(x)

        if SSqr == 0:
            EI = 0
        else:
            EI = (self.y_best - y_hat) * (0.5 + 0.5*erf((self.y_best - y_hat)/np.sqrt(2 * SSqr))) + \
                    np.sqrt(0.5*SSqr/np.pi)*np.exp(-0.5*(self.y_best - y_hat)**2/SSqr)
            EI = -EI

        return EI
