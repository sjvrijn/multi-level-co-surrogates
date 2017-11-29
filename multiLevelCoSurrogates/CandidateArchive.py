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


    def addcandidate(self, candidate, fitness, fidelity=None):
        """"""

        if self.num_fidelities == 1 and fidelity is not None:
            raise Warning(f"fidelity specification {fidelity} ignored in single-fidelity case")
        elif self.num_fidelities > 1 and fidelity is None:
            raise ValueError('must specify fidelity level in multi-fidelity case')

        if type(fidelity) in [tuple, list]:
            # TODO: Implement case for adding multiple fitnesses at the same time
            raise NotImplementedError

        if self.num_fidelities == 1:
            fit_values = [fitness]
        else:
            fidelity = f'fitness_{fidelity}'
            fit_values = [np.nan] * self.num_fidelities
            idx = self.fidelities.index(fidelity)
            fit_values[idx] = fitness

        row = np.array([candidate, fit_values]).flatten()
        self.data.loc[len(self.data)] = row


    def updatecandidate(self, candidate, fitness, fidelity=None, *, verbose=False):
        """"""

        query_parts = [f'x{i} == {val}' for i, val in enumerate(candidate)]
        idx = self.data.query(' & '.join(query_parts)).iloc[-1].name

        if fidelity is None:
            fidelity = 'fitness'
        else:
            fidelity = f'fitness_{fidelity}'

        if not np.isnan(self.data.at[idx, fidelity]) and verbose:
            raise Warning(f"overwriting existing value '{self.data[idx, fidelity]}' with '{fitness}'")

        self.data.at[idx, fidelity] = fitness
