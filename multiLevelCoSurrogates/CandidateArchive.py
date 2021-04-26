#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
CandidateArchive.py: Class to store candidate solutions in an optimization process with their respective
                     (multi-fidelity) fitness values
"""
from collections import namedtuple
from typing import Dict, Iterable, Union

from more_itertools import pairwise

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


import numpy as np
from warnings import warn

from multiLevelCoSurrogates.Utils import BiFidelityDoE

CandidateSet = namedtuple('CandidateSet', ['candidates', 'fitnesses'])


class CandidateArchive:

    def __init__(self, ndim: int, fidelities=None):
        """An archive of candidate: fitnesses pairs, for one or multiple fidelities"""
        self.ndim = ndim

        if not fidelities:
            fidelities = ['fitness']
        self.fidelities = fidelities

        self.data = {}
        self.max = {fid: -np.inf for fid in self.fidelities}
        self.min = {fid: np.inf for fid in self.fidelities}


    @classmethod
    def from_multi_fidelity_function(cls, multi_fid_func, ndim: int=None):
        """Create a CandidateArchive based on a multi-fidelity function. This
        creates an archive with 'columns' for every fidelity of the function
        *plus* a difference-column for each consequtive pair of fidelities.

        Dimensionality `ndim` is taken from `func.ndim` by default, but can be
        overwritten manually.

        Examples:
            For `fidelities=('high', 'low')`, creates an archive with
            `fidelities=('high', 'low', 'high-low')`

            For `fidelities=('high', 'medium', 'low')`, creates an archive with
            `fidelities=('high', 'medium', 'low', 'high-medium', 'medium-low')`
        """
        if ndim is None:
            ndim = multi_fid_func.ndim

        diff_fidelities = tuple('-'.join(fidelities)
                                for fidelities in pairwise(multi_fid_func.fidelity_names))

        fidelities = tuple(multi_fid_func.fidelity_names) + diff_fidelities

        return cls(ndim=ndim, fidelities=fidelities)


    def __contains__(self, val):
        return tuple(val) in self.data


    def __len__(self):
        return len(self.data)


    def __index__(self, val) -> dict[str: float]:
        return {
            fid: fitness
            for fid, fitness
            in zip(self.fidelities, self.data[tuple(val)])
        }


    def addcandidates(self, candidates, fitnesses, fidelity: str=None, *, verbose: bool=False):
        """Add multiple candidates to the archive"""
        for cand, fit in zip(candidates, fitnesses):
            self.addcandidate(cand, fit, fidelity=fidelity, verbose=verbose)


    def addcandidate(self, candidate, fitness, fidelity: str=None, *, verbose: bool=False):
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


    def getfitnesses(self, candidates: Iterable, fidelity: Union[str, Iterable[str]]=None) -> Iterable:
        """Return the relevant fitness values for the given candidates"""

        fitnesses = np.array([
            self.data[tuple(candidate)]
            for candidate in candidates
        ])

        if isinstance(fidelity, str):
            return fitnesses[:,self.fidelities.index(fidelity)]
        elif fidelity:
            return np.hstack([
                fitnesses[:, self.fidelities.index(fid)].reshape(-1, 1)
                for fid in fidelity
            ])

        return fitnesses

    def _addnewcandidate(self, candidate, fitness, fidelity: str=None, *, verbose: bool=False):
        if len(self.fidelities) == 1:
            fit_values = [fitness]
        else:
            fit_values = np.array([np.nan] * len(self.fidelities))
            idx = self.fidelities.index(fidelity)
            fit_values[idx] = fitness

        self._updateminmax(fidelity, fitness)
        self.data[tuple(candidate)] = fit_values


    def _updatecandidate(self, candidate, fitness, fidelity: str=None, *, verbose: bool=False):
        fit_values = self.data[tuple(candidate)]

        if fidelity is None:
            fidelity = 'fitness'

        fid_idx = self.fidelities.index(fidelity)

        if verbose and not np.isnan(fit_values[fid_idx]):
            warn(f"overwriting existing value '{self.data[tuple(candidate), fid_idx]}' with '{fitness}'", RuntimeWarning)

        fit_values[fid_idx] = fitness
        self._updateminmax(fidelity, fitness)


    def getcandidates(self, fidelity: str=None, num_recent_candidates: int=None):
        """Retrieve candidates and fitnesses from the archive.

        :param fidelity:                (optional) Only return candidate and fitness information for the specified fidelities
        :param num_recent_candidates:   (optional) Only return the last `n` candidates added to the archive
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


    def as_doe(self):
        """Present the stored candidates as a bi-fidelity DoE"""
        return BiFidelityDoE(
            self.getcandidates(fidelity='high').candidates,
            self.getcandidates(fidelity='low').candidates,
        )


    def count(self, fidelity: str='fitness'):
        """Count the number of samples archived for the given fidelity"""
        if not fidelity:
            return len(self.data)

        idx = self.fidelities.index(fidelity)
        return sum(
            1
            for fitnesses in self.data.values()
            if not np.isnan(fitnesses[idx])
        )


    def _updateminmax(self, fidelity: str, value):
        if value > self.max[fidelity]:
            self.max[fidelity] = value
        elif value < self.min[fidelity]:
            self.min[fidelity] = value
