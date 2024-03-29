#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
candidate_archive_new.py: Reimplementation of CandidateArchive to include indices,
                          be nicer and easier.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


from collections import namedtuple
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Iterable, Union
from warnings import warn

import numpy as np

import mf2
from multiLevelCoSurrogates.utils import idx_set, split_set, BiFidelityDoE, LowHighFidSamplesWarning, NoHighFidTrainSamplesWarning, NoSpareLowFidSamplesWarning

CandidateSet = namedtuple('CandidateSet', ['candidates', 'fitnesses'])


class CandidateArchive:

    def __init__(self, *args, **kwargs):
        """Archive of candidates that record fitness in multiple fidelities"""
        self.candidates = []
        self._update_history = []
        self._all_fidelities = {}  # dictionary keys used as 'ordered set'

        self.min = {}
        self.max = {}


    @classmethod
    def from_multi_fidelity_function(cls, multi_fid_func: mf2.MultiFidelityFunction):
        return cls()


    @classmethod
    def from_bi_fid_doe(cls, high_x: np.ndarray, low_x: np.ndarray,
                        high_y: Iterable[float], low_y: Iterable[float]):
        """Create a populated CandidateArchive from an existing bi-fidelity
        Design of Experiments (DoE) [high_x, low_x] with corresponding fitness
        values [high_y, low_y]
        """
        archive = cls()
        archive.addcandidates(low_x, low_y, fidelity='low')
        archive.addcandidates(high_x, high_y, fidelity='high')
        return archive


    @classmethod
    def from_file(cls, file: Union[str, Path]):
        """Load a candidate archive from a previously saved .npz file"""
        loaded_data = np.load(file)
        candidates = loaded_data['candidates']
        archive = cls()

        for fid in loaded_data['fidelities']:
            fitnesses = loaded_data[fid]

            # NaN values are stored in file for completeness,
            # but must be excluded again when adding
            nan_fitnesses = np.isnan(fitnesses)
            archive.addcandidates(
                candidates[~nan_fitnesses], fitnesses[~nan_fitnesses],
                fidelity=fid
            )

        # Overwrite history
        archive._update_history = loaded_data['history'].tolist()
        return archive


    @property
    def fidelities(self):
        return list(self._all_fidelities.keys())


    @property
    def indices(self):
        return tuple(sorted(c.idx for c in self.candidates if 'high' in c.fidelities)),\
               tuple(sorted(c.idx for c in self.candidates if 'low' in c.fidelities))


    def addcandidates(self, candidates: np.ndarray,
                      fitnesses: Iterable[float], fidelity: str=None):
        """Add candidate, fitness pairs to the archive for given fidelity.
        Will overwrite fitness value if already present.
        """
        for candidate, fitness in zip(candidates, np.ravel(fitnesses)):
            self.addcandidate(candidate, fitness, fidelity)


    def addcandidate(self, candidate: np.ndarray,
                     fitness: Union[float, np.ndarray], fidelity: str=None):
        """Add a candidate, fitness pair to the archive for given fidelity.
        Will overwrite fitness value if already present
        """
        # unspecified fidelity when explicit fidelities are present raises Error
        if fidelity is None:
            if len(self.fidelities) > 1:
                raise ValueError(f'Since explcit fidelities are present, new candidates '
                                 f'cannot be added with implicit fidelity. Fidelities '
                                 f'currently present: {self.fidelities}')
            elif len(self.fidelities) == 1:
                fidelity = self.fidelities[0]
            else:
                fidelity = '_fitness_'

        # explicitly casting to float in case it isn't a number, e.g. np.ndarray
        if not isinstance(fitness, Number):
            fitness = float(fitness)

        if np.isnan(fitness):
            raise ValueError(f'invalid fitness value `NaN` for candidate {candidate}')

        try:
            idx = self.candidates.index(candidate)
            # ignore update if re-adding same fitness value
            if np.isclose(self.candidates[idx].fidelities.get(fidelity, np.nan), fitness):
                return

            self.candidates[idx].fidelities[fidelity] = fitness

        except ValueError:
            # candidates is not yet present
            idx = len(self.candidates)
            fidelities = {fidelity: fitness}
            self.candidates.append(Candidate(idx, candidate, fidelities))

        self._update_history.append((idx, fidelity))  # record update order
        self._all_fidelities[fidelity] = None  # add key entry
        self._updateminmax(fitness, fidelity)


    def getfitnesses(self, candidates: np.ndarray,
                     fidelity: Union[str, Iterable[str]]) -> np.ndarray:
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


    def getcandidates(self, fidelity: Union[str, Iterable[str]]=None,
                      num_recent_candidates: int=None) -> CandidateSet:
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

            # add if all specified fidelities are present for this candidate
            if not np.count_nonzero(np.isnan(fitness)):
                fitnesses.append(fitness)
                candidates.append(candidate.x)

        return CandidateSet(np.array(candidates), np.array(fitnesses))


    def as_doe(self) -> BiFidelityDoE:
        """Present the stored candidates as a bi-fidelity DoE"""
        return BiFidelityDoE(
            self.getcandidates(fidelity='high').candidates,
            self.getcandidates(fidelity='low').candidates,
        )


    def count(self, fidelity: str=None) -> int:
        """Count the number of samples archived for the given fidelity"""
        return sum(
            fidelity in candidate.fidelities
            for candidate in self.candidates
        )


    def save(self, file: Union[str, Path]):
        """Save the archive as a reusable .npz file

        Stores candidates, fitness values per fidelity and the update history,
        allowing for a perfect recreation of the archive at a later date.
        """
        # prepare a type specification to correctly store the update_history
        history_type = np.dtype([
            ('idx', int),
            ('fidelity', np.unicode_, max(len(str(f)) for f in self.fidelities)),
        ])

        save_args = {
            'candidates': np.array([c.x for c in self.candidates]),
            'history': np.array(self._update_history, dtype=history_type),
            'fidelities': [str(f) for f in self.fidelities],
        }
        for fid in self.fidelities:
            fitnesses = self.getfitnesses(save_args['candidates'], fidelity=fid)
            save_args[fid] = fitnesses

        np.savez(file, **save_args)


    def undo_last(self, *, fidelity=None):
        """Undo the latest addition to the archive

        This only undoes additions. If a value has been first added and then
        overwritten, `undo_last()` will simply remove it.
        Undone actions also cannot be replayed, so be sure to have an original
        copy (saved) and use with care.
        """
        # If no fidelity is specified, simply select the last one to undo
        if fidelity is None:
            fidelity = self._update_history[-1][1]

        if fidelity not in self.fidelities:
            raise ValueError(f'Specified fidelity `{fidelity}` is not in archive.')

        while True:  # do-while
            idx, fid = self._update_history.pop(-1)
            del self.candidates[idx].fidelities[fid]

            if fid == fidelity:
                break


    def split(self, num_high: int, num_low: int) -> tuple['CandidateArchive', 'CandidateArchive']:
        r"""Split the archive into two according to the given size.

        Assumes the archive has fidelity levels 'high' and 'low'. Will split
        such that the first returned archive matches the specified size, and the
        other archive contains all remaining samples and fitness values.

        Illustrative example of splitting an archive with 3 high-fidelity and 5
        low-fidelity into one with 2 high- and 3 low-fidelity, and the remainder
        (1 high-, 2 low-fidelity).

                                    Original archive:

                               | candidate | high | low |
                               |-----------|------|-----|
                               | [0.0,0.1] |  x   |  x  |
                               | [0.2,0.3] |  x   |  x  |
                               | [0.4,0.5] |      |  x  |
                               | [0.6,0.7] |      |  x  |
                               | [0.8,0.9] |  x   |  x  |

                                        /        \
                                     /              \
                                  /                    \
                               /                          \
        Selected            /                                \        Remainder

        | candidate | high | low |                    | candidate | high | low |
        |-----------|------|-----|                    |-----------|------|-----|
        | [0.0,0.1] |      |  x  |                    | [0.0,0.1] |  x   |     |
        | [0.2,0.3] |  x   |  x  |                    | [0.2,0.3] |      |     |
        | [0.4,0.5] |      |     |                    | [0.4,0.5] |      |  x  |
        | [0.6,0.7] |      |     |                    | [0.6,0.7] |      |  x  |
        | [0.8,0.9] |  x   |  x  |                    | [0.8,0.9] |      |     |

        Note that 'high' -> 'low'                     Note that 'high' !-> 'low'

        Note: this method only makes deep copies of the Candidates when needed,
        so any changes in the original archive may apply to the split archives
        and vice versa.
        """
        cur_num_high, cur_num_low = self.count('high'), self.count('low')
        # Errors
        if 'high' not in self.fidelities or 'low' not in self.fidelities:
            raise ValueError("Fidelity levels 'high' and 'low' are both "
                             "required, but not present")
        if not 0 <= num_high <= cur_num_high:
            raise ValueError(f"'num_high' must be in the range [0, len(doe.high) "
                             f"(={cur_num_high})], but is {num_high}")
        if num_low > cur_num_low:
            raise ValueError(f"'num_low' cannot be greater than len(doe.low) "
                             f"(={cur_num_low}), but is {num_low}")
        if num_low < num_high:
            raise ValueError(f"'num_low' must be at least 'num_high', "
                             f"but {num_low} < {num_high}")

        # Warnings
        if num_high < 2:
            warn("Not enough high-fidelity samples selected to serve as a training set",
                 category=LowHighFidSamplesWarning)
        if num_high == cur_num_high:
            warn("All high-fidelity samples selected, none left over as test set",
                 category=NoHighFidTrainSamplesWarning)
        if num_low == num_high:
            warn("No additional low-fidelity samples to be selected",
                 category=NoSpareLowFidSamplesWarning)

        selected = CandidateArchive()
        other = CandidateArchive()

        # prepare the sets of indices by which to split the candidates
        high_indices = {idx for idx, c in enumerate(self.candidates) if 'high' in c.fidelities}
        high_select, high_other = split_set(high_indices, num_high)
        # select remaining low-fidelity from those not already included with high-fidelity
        low_leftover = idx_set(self.candidates) - high_select
        low_select, low_other = split_set(low_leftover, num_low - num_high)

        # high-fidelity candidates for which only low-fidelity is selected must be split
        if to_split_up := low_select & high_other:
            low_select -= to_split_up
            high_other -= to_split_up
            split_candidates = [self.candidates[idx].split() for idx in to_split_up]
            for high, low in split_candidates:
                selected.candidates.append(low)
                other.candidates.append(high)

        # add the relevant candidates
        selected.candidates.extend([self.candidates[idx] for idx in high_select | low_select])
        other.candidates.extend([self.candidates[idx] for idx in high_other | low_other])

        return selected, other


    def _updateminmax(self, value: float, fidelity: str=None):

        if fidelity not in self.max:  # or self.min
            self.max[fidelity] = self.min[fidelity] = value
        elif value > self.max[fidelity]:
            self.max[fidelity] = value
        elif value < self.min[fidelity]:
            self.min[fidelity] = value


    def __len__(self):
        return len(self.candidates)


    def __contains__(self, item):
        # comparison order matters: want to use candidate.__eq__
        return any(candidate == item for candidate in self.candidates)


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

    def split(self) -> tuple['Candidate', 'Candidate']:
        """Split the candidate into two copies with only high- or low-fidelity"""
        return Candidate(self.idx, self.x, {'high': self.fidelities['high']}), \
               Candidate(self.idx, self.x, {'low': self.fidelities['low']})
