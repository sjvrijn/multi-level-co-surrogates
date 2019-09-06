#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
CandidateArchive.py: Class to store candidate solutions in an optimization process with their respective
                     (multi-fidelity) fitness values
"""
from collections import namedtuple

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


import numpy as np
from warnings import warn


CandidateSet = namedtuple('CandidateSet', ['candidates', 'fitnesses'])


class CandidateArchive:

    def __init__(self, ndim, fidelities=None):
        """An archive of candidate: fitnesses pairs, for one or multiple fidelities"""
        self.ndim = ndim

        if not fidelities:
            fidelities = ['fitness']
        self.fidelities = fidelities

        self.data = {}
        self.max = {fid: -np.inf for fid in self.fidelities}
        self.min = {fid: np.inf for fid in self.fidelities}


    def __len__(self):
        return len(self.data)


    def addcandidates(self, candidates, fitnesses, fidelity=None, *, verbose=False):
        """Add multiple candidates to the archive"""
        for cand, fit in zip(candidates, fitnesses):
            self.addcandidate(cand, fit, fidelity=fidelity, verbose=verbose)


    def addcandidate(self, candidate, fitness, fidelity=None, *, verbose=False):
        """Add a candidate to the archive. Will overwrite fitness value if candidate is already present"""

        if len(self.fidelities) == 1 and fidelity is not None and verbose:
            warn(f"fidelity specification {fidelity} ignored in single-fidelity case", RuntimeWarning)
        elif len(self.fidelities) > 1 and fidelity is None:
            raise ValueError('must specify fidelity level in multi-fidelity case')

        if fidelity is None:
            fidelity = self.fidelities

        # Checking types to make sure they are iterable in the right way
        if isinstance(fitness, (np.float64, float)):
            fitness = [fitness]

        if isinstance(fidelity, str):
            fidelity = [fidelity]

        for fid, fit in zip(fidelity, list(fitness)):
            if tuple(candidate) not in self.data:
                self._addnewcandidate(candidate, fit, fid, verbose=verbose)
            else:
                self._updatecandidate(candidate, fit, fid, verbose=verbose)


    def _addnewcandidate(self, candidate, fitness, fidelity=None, *, verbose=False):
        if len(self.fidelities) == 1:
            fit_values = [fitness]
        else:
            fit_values = np.array([np.nan] * len(self.fidelities))
            idx = self.fidelities.index(fidelity)
            fit_values[idx] = fitness

        self._updateminmax(fidelity, fitness)
        self.data[tuple(candidate)] = fit_values


    def _updatecandidate(self, candidate, fitness, fidelity=None, *, verbose=False):
        fit_values = self.data[tuple(candidate)]

        if fidelity is None:
            fidelity = 'fitness'

        fid_idx = self.fidelities.index(fidelity)

        if verbose and not np.isnan(fit_values[fid_idx]):
            warn(f"overwriting existing value '{self.data[tuple(candidate), fid_idx]}' with '{fitness}'", RuntimeWarning)

        fit_values[fid_idx] = fitness
        self._updateminmax(fidelity, fitness)


    def getcandidates(self, num_recent_candidates=None, fidelity=None):
        """Retrieve candidates and fitnesses from the archive.

        :param num_recent_candidates:   (optional) Only return the last `n` candidates added to the archive
        :param fidelity:                (optional) Only return candidate and fitness information for the specified fidelities
        :return:                        Candidates, Fitnesses (tuple of numpy arrays)
        """

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
            else:
                candidates.append(list(candidate))
                fitnesses.append([fits[idx] for idx in indices])

        candidates = np.array(candidates)
        fitnesses = np.array(fitnesses)

        if num_recent_candidates is not None:
            candidates = candidates[-num_recent_candidates:]
            fitnesses = fitnesses[-num_recent_candidates:]

        return CandidateSet(candidates, fitnesses)


    def _updateminmax(self, fidelity, value):
        if value > self.max[fidelity]:
            self.max[fidelity] = value
        elif value < self.min[fidelity]:
            self.min[fidelity] = value
