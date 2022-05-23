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


    def addcandidates(self, candidates, fitnesses, fidelity=None):
        """Add candidate, fitness pairs to the archive for given fidelity.
        Will overwrite fitness value if already present.
        """
        for candidate, fitness in zip(candidates, fitnesses):
            self.addcandidate(candidate, fitness, fidelity)


    def addcandidate(self, candidate, fitness, fidelity=None):
        """Add a candidate, fitness pair to the archive for given fidelity.
        Will overwrite fitness value if already present
        """
        try:
            idx = self.candidates.index(candidate)
            self.candidates[idx].fidelities[fidelity] = fitness

        except ValueError:
            # candidates is not yet present
            new_idx = len(self.candidates)
            fidelities = {fidelity: fitness}
            self.candidates.append(Candidate(new_idx, candidate, fidelities))


    def getfitnesses(self, candidates: Iterable, fidelity: Union[str, Iterable[str]]=None) -> Iterable:
        """Return the relevant fitness values for the given candidates"""

        # retrieve all candidates by index, fails if candidate is not in archive
        candidate_indices = [
            self.candidates.index(candidate)
            for candidate in candidates
        ]
        fidelities = [
            self.candidates[i].fidelities
            for i in candidate_indices
        ]

        if isinstance(fidelity, str):
            return np.array([
                fid.get(fidelity, np.nan)
                for fid in fidelities
            ])

        elif fidelity:
            return np.array([
                [fid.get(f, np.nan) for f in fidelity]
                for fid in fidelities
            ])

        else:
            # can probably be cached upon adding candidates...
            all_fidelities = set()
            for fid in fidelities:
                all_fidelities.update(fid.keys())
            all_fidelities = list(all_fidelities)

            return np.array([
                [fid.get(f, np.nan) for f in all_fidelities]
                for fid in fidelities
            ])


    def getcandidates(self, *args, **kwargs):
        """Retrieve candidates and fitnesses from the archive.

        :param fidelity:                (optional) Only return candidate and fitness information for the specified fidelities
        :param num_recent_candidates:   (optional) Only return the last `n` candidates added to the archive
        :return:                        Candidates, Fitnesses (tuple of numpy arrays)
        """
        pass


    def as_doe(self):
        """Present the stored candidates as a bi-fidelity DoE"""
        pass


    def count(self, fidelity: str=None):
        """Count the number of samples archived for the given fidelity"""
        return sum(
            1
            for candidate in self.candidates
            if fidelity in candidate.fidelities
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
