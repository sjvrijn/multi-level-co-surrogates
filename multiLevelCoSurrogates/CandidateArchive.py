#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals


'''
CandidateArchive.py: Class to store candidate solutions in an optimization process with their respective
                     (multi-fidelity) fitness values
'''

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


import numpy as np
from warnings import warn


#TODO: add support for automatic difference columns
class CandidateArchive:

    def __init__(self, ndim, fidelities=None):
        """"""
        self.ndim = ndim

        if fidelities:
            self.num_fidelities = len(fidelities)
            self.fidelities = fidelities
        else:
            self.num_fidelities = 1
            self.fidelities = ['fitness']

        self.data = {}


    def __len__(self):
        return len(self.data)


    def addcandidate(self, candidate, fitness, fidelity=None, *, verbose=False):
        """"""

        if self.num_fidelities == 1 and fidelity is not None:
            warn(f"fidelity specification {fidelity} ignored in single-fidelity case", RuntimeWarning)
        elif self.num_fidelities > 1 and fidelity is None:
            raise ValueError('must specify fidelity level in multi-fidelity case')

        if type(fidelity) in [tuple, list]:
            # TODO: Implement case for adding multiple fitnesses at the same time
            raise NotImplementedError

        if self.num_fidelities == 1:
            fit_values = fitness
        else:
            fit_values = np.array([np.nan] * self.num_fidelities)
            idx = self.fidelities.index(fidelity)
            fit_values[idx] = fitness

        self.data[tuple(candidate)] = fit_values


    def updatecandidate(self, candidate, fitness, fidelity=None, *, verbose=False):
        """"""

        fit_values = self.data[tuple(candidate)]

        if fidelity is None:
            fidelity = 'fitness'

        fid_idx = self.fidelities.index(fidelity)

        if verbose and not np.isnan(fit_values[fid_idx]):
            warn(f"overwriting existing value '{self.data[idx, fidelity]}' with '{fitness}'", RuntimeWarning)

        fit_values[fid_idx] = fitness


    def getcandidates(self, n=None, fidelity=None):
        """"""

        if type(fidelity) in [tuple, list]:
            pass
        elif fidelity:
            fidelity = [fidelity]
        else:
            fidelity = ['fitness']

        indices = [self.fidelities.index(fid) for fid in fidelity]

        candidates = []
        fitnesses = []
        for candidate, fits in self.data.items():
            for idx in indices:
                if np.isnan(fits[idx]):
                    break
            candidates.append(list(candidate))
            fitnesses.append([fits[idx] for idx in indices])

        candidates = np.array(candidates)
        fitnesses = np.array(fitnesses)

        if n is not None:
            candidates = candidates[-n:]
            fitnesses = fitnesses[-n:]

        return candidates, fitnesses
