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
import pandas as pd


class CandidateArchive:

    def __init__(self, ndim, fidelities=None):
        """"""
        self.ndim = ndim

        if fidelities:
            self.num_fidelities = len(fidelities)
            self.fidelities = [f'fitness_{fid}' for fid in fidelities]
        else:
            self.num_fidelities = 1
            self.fidelities = ['fitness']

        columns = [f'x{i}' for i in range(ndim)]
        columns += self.fidelities

        self.data = pd.DataFrame(columns=columns)


    def __len__(self):
        return len(self.data)


    def _getcandidateindex(self, candidate):
        """"""
        query_parts = [f'x{i} == {val}' for i, val in enumerate(candidate)]
        idx = self.data.query(' & '.join(query_parts)).iloc[-1].name
        return idx


    def addcandidate(self, candidate, fitness, fidelity=None):
        """"""

        if self.num_fidelities == 1 and fidelity is not None:
            raise Warning(f"fidelity specification {fidelity} ignored in single-fidelity case")
        elif self.num_fidelities > 1 and fidelity is None:
            raise ValueError('must specify fidelity level in multi-fidelity case')

        if type(fidelity) in [tuple, list]:
            # TODO: Implement case for adding multiple fitnesses at the same time
            raise NotImplementedError

        try:  # Check if candidate already exists
            _ = self._getcandidateindex(candidate)
            raise ValueError(f"candidate {candidate} is already present in the archive."
                             f" Use 'CandidateArchive.updatecandidate()' instead")
        except IndexError:
            pass  # candidate does not yet exist, we'll add the new one as intended

        if self.num_fidelities == 1:
            fit_values = fitness
        else:
            fidelity = f'fitness_{fidelity}'
            fit_values = np.array([np.nan] * self.num_fidelities)
            idx = self.fidelities.index(fidelity)
            fit_values[idx] = fitness

        row = np.hstack((candidate, fit_values)).flatten()
        self.data.loc[len(self.data)] = row


    def updatecandidate(self, candidate, fitness, fidelity=None, *, verbose=False):
        """"""

        idx = self._getcandidateindex(candidate)

        if fidelity is None:
            fidelity = 'fitness'
        else:
            fidelity = f'fitness_{fidelity}'

        if not np.isnan(self.data.at[idx, fidelity]) and verbose:
            raise Warning(f"overwriting existing value '{self.data[idx, fidelity]}' with '{fitness}'")

        self.data.at[idx, fidelity] = fitness


    def getcandidates(self, n=None, fidelity=None):
        """"""

        if type(fidelity) in [tuple, list]:
            fidelity = [f'fitness_{fid}' for fid in fidelity]
        elif fidelity:
            fidelity = [f'fitness_{fidelity}']
        else:
            fidelity = ['fitness']

        selected_data = self.data
        for fid in fidelity:
            selected_data = selected_data[selected_data[fid].notnull()]

        candidates = selected_data.as_matrix(columns=self.data.columns[:self.ndim])
        fitnesses = selected_data.as_matrix(columns=fidelity)

        if n is not None:
            candidates = candidates[-n:]
            fitnesses = fitnesses[-n:]

        return candidates, fitnesses
