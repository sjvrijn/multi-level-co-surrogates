#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Class file for a multi-fidelity model based on hierarchical surrogate models
"""

from more_itertools import stagger

from multiLevelCoSurrogates import CandidateArchive
from .Surrogates import HierarchicalSurrogate, Surrogate


class MultiFidelityModel:

    def __init__(
            self, fidelities, archive: CandidateArchive=None, *,
            normalized=True, surrogate_name='Kriging', kernel=None, scaling='on'
    ):
        """A complete multi-fidelity model consisting of an arbitrary amount of
         stacked hierarchical surrogate models with a regular surrogate model
         down at the bottom of the stack.

        :param fidelities:      List of fidelity names, from highest to lowest, excluding differences
        :param archive:         Archive containing all samples to train the models on
        :param normalized:
        :param surrogate_name:
        :param kernel:
        :param scaling:
        """
        self.fidelities = fidelities
        self.archive = archive
        self.normalized = normalized

        kwargs = {
            'surrogate_name': surrogate_name,
            'candidate_archive': self.archive,
            'normalized': normalized,
            'kernel': kernel
        }

        self.models = {
            fidelities[-1]: Surrogate.fromname(**kwargs, fidelity=fidelities[-1])
        }

        for fid_low, fid_high in stagger(reversed(self.fidelities), offsets=(0, 1)):
            model = HierarchicalSurrogate(
                **kwargs,
                fidelities=[fid_high, fid_low],
                lower_fidelity_model=self.models[fid_low],
                scaling=scaling,
            )
            self.models[fid_high] = model

        self.top_level_model = self.models[self.fidelities[0]]
        self.retrain()


    def retrain(self):
        self.top_level_model.retrain()
