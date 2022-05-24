#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
CandidateArchiveNew.py: Reimplementation of CandidateArchive to include indices,
                        be nicer and easier.
"""
from collections import namedtuple
from dataclasses import dataclass
from itertools import chain
from typing import Iterable, Union

from more_itertools import pairwise

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


import numpy as np
from warnings import warn

import mf2
from multiLevelCoSurrogates.utils import BiFidelityDoE
from .CandidateArchive import CandidateSet


class CandidateArchiveNew:

    def __init__(self, *args, **kwargs):
        """Archive of candidates that record fitnessin multiple fidelities"""
        self.candidates = []
        self._update_history = []
        self._all_fidelities = {}  # dictionary keys used as 'ordered set'


    @classmethod
    def from_multi_fidelity_function(
            cls,
            multi_fid_func: mf2.MultiFidelityFunction,
            *args,
            **kwargs
    ):

        return cls()


    @classmethod
    def from_bi_fid_DoE(cls, high_x, low_x, high_y, low_y):
        """Create a populated CandidateArchive from an existing bi-fidelity DoE
        (high_x, low_x) with corresponding fitness values (high_y, low_y)
        """
        archive = cls()
        archive.addcandidates(low_x, low_y, fidelity='low')
        archive.addcandidates(high_x, high_y, fidelity='high')
        return archive


    @property
    def fidelities(self):
        return self._all_fidelities.keys()


    def addcandidates(self, candidates, fitnesses, fidelity=None):
        """Add candidate, fitness pairs to the archive for given fidelity.
        Will overwrite fitness value if already present.
        """
        for candidate, fitness in zip(candidates, np.ravel(fitnesses)):
            self.addcandidate(candidate, fitness, fidelity)


    def addcandidate(self, candidate, fitness, fidelity=None):
        """Add a candidate, fitness pair to the archive for given fidelity.
        Will overwrite fitness value if already present
        """
        # unspecified fidelity when explicit fidelities are present raises Error
        if fidelity is None and len(self.fidelities) >= 1 and fidelity not in self.fidelities:
            raise ValueError(f'Since explcit fidelities are present, new candidates '
                             f'cannot be added with implicit fidelity. Fidelities '
                             f'currently present: {self.fidelities}')

        try:
            idx = self.candidates.index(candidate)
            self.candidates[idx].fidelities[fidelity] = fitness

        except ValueError:
            # candidates is not yet present
            new_idx = len(self.candidates)
            fidelities = {fidelity: fitness}
            self.candidates.append(Candidate(new_idx, candidate, fidelities))

        # create key entry if it does not yet exist
        self._all_fidelities[fidelity] = None


    def getfitnesses(self, candidates: Iterable, fidelity: Union[str, Iterable[str]]) -> Iterable:
        """Return the relevant fitness values for the given candidates"""

        if isinstance(fidelity, str):
            fidelity = [fidelity]

        # retrieve all candidates by index, fails if candidate is not in archive
        candidate_indices = [
            self.candidates.index(candidate)
            for candidate in candidates
        ]

        fitnesses = np.array([
            [self.candidates[i].fidelities.get(f, np.nan) for f in fidelity]
            for i in candidate_indices
        ])

        if len(fidelity) == 1:
            return fitnesses.reshape(-1)
        return fitnesses


    def getcandidates(self, fidelity=None):
        """Retrieve candidates and fitnesses from the archive.
        :fidelity:  (List of) fidelities to select by. Default: all
        :return:    Candidates, Fitnesses (tuple of numpy arrays)
        """
        if not fidelity:
            fidelity = self.fidelities
        elif isinstance(fidelity, str):
            fidelity = [fidelity]

        candidates, fitnesses = [], []
        for candidate in self.candidates:
            fitness = [
                candidate.fidelities.get(fid, np.nan)
                for fid in fidelity
            ]

            # add if any specified fidelities are present for this candidate
            if np.count_nonzero(~np.isnan(fitness)):
                fitnesses.append(fitness)
                candidates.append(candidate.x)

        return CandidateSet(np.array(candidates), np.array(fitnesses))


    def as_doe(self):
        """Present the stored candidates as a bi-fidelity DoE"""
        return BiFidelityDoE(
            self.getcandidates(fidelity='high').candidates,
            self.getcandidates(fidelity='low').candidates,
        )


    def count(self, fidelity: str=None):
        """Count the number of samples archived for the given fidelity"""
        return sum(
            fidelity in candidate.fidelities
            for candidate in self.candidates
        )


    def _addnewcandidate(self, *args, **kwargs):
        pass


    def _updatecandidate(self, *args, **kwargs):
        pass


    def _updateminmax(self, value: float, fidelity: str=None):
        # if value > self.max[fidelity]:
        #     self.max[fidelity] = value
        # elif value < self.min[fidelity]:
        #     self.min[fidelity] = value
        pass


    def __contains__(self, val):
        pass


    def __len__(self):
        return len(self.candidates)


    def __index__(self, val):
        pass


@dataclass(eq=False)
class Candidate:
    idx: int
    x: np.ndarray
    fidelities: dict[str, float]

    def __eq__(self, other):
        """Check equality only by checking Candidate.x"""
        if isinstance(other, Candidate):
            other = other.x
        return np.all(self.x == other)
