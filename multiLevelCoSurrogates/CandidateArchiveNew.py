#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
CandidateArchiveNew.py: Reimplementation of CandidateArchive to include indices,
                        be nicer and easier.
"""
from collections import namedtuple
from dataclasses import dataclass
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


    def addcandidates(self, *args, **kwargs):
        """Add multiple candidates to the archive"""
        pass


    def addcandidate(self, candidate, fitness, fidelity=None):
        """Add a candidate to the archive.
        Will overwrite fitness value if candidate is already present
        """
        try:
            idx = self.candidates.index(candidate)
            self.candidates[idx].fidelities[fidelity] = fitness

        except ValueError:
            # candidates is not yet present
            new_idx = len(self.candidates)
            fidelities = {fidelity: fitness}
            self.candidates.append(Candidate(new_idx, candidate, fidelities))


    def getfitnesses(self, *args, **kwargs):
        """Return the relevant fitness values for the given candidates"""
        pass


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


    def count(self, *args, **kwargs):
        """Count the number of samples archived for the given fidelity"""
        pass


    def _addnewcandidate(self, *args, **kwargs):
        pass


    def _updatecandidate(self, *args, **kwargs):
        pass


    def _updateminmax(self, *args, **kwargs):
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
