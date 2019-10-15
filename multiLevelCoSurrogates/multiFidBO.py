#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Class file for a multi-fidelity Bayesian Optimizer with built-in hierarchical
surrogate model
"""

import numpy as np
import pandas as pd
import bayes_opt as bo

from functools import partial
from collections import namedtuple
from sklearn.metrics import mean_squared_error, r2_score
from more_itertools import pairwise, stagger

from .CandidateArchive import CandidateArchive
from .Utils import create_random_sample_set, rescale, \
    sample_by_function, low_lhs_sample, \
    createsurfaces, plotsurfaces, gpplot, ScatterPoints, ValueRange
from .Surrogates import HierarchicalSurrogate, Surrogate



class MultiFidelityBO:

    def __init__(self, multi_fid_func, archive=None, save_plot=False, show_plot=False,
                 schema=None, normalized=True, test_sample=None, minimize=False,
                 kernel=None, scaling='on'):

        if minimize:
            raise NotImplementedError("Minimization is not internally supported. "
                                      "Instead, please invert your function(s).")

        self.show_plot = show_plot
        self.save_plot = save_plot
        self.func = multi_fid_func
        if archive:
            self.ndim = archive.ndim
        else:
            self.ndim = self.func.ndim

        if len(self.func.u_bound) == self.ndim and len(self.func.l_bound) == self.ndim:
            self.bounds = np.array([self.func.l_bound, self.func.u_bound], dtype=np.float)
        elif len(self.func.u_bound) == 1 and len(self.func.u_bound) == 1:
            self.bounds = np.array([self.func.l_bound * self.ndim, self.func.u_bound * self.ndim], dtype=np.float)
        else:
            raise ValueError(f"Unclear how to extend lower/upper bound into {self.ndim} dimensions")

        self.input_range = ValueRange(*self.bounds)
        self.fidelities = list(self.func.fidelity_names)
        self.normalized = normalized

        if schema is None:
            self.schema = list(reversed([2**i for i in range(len(self.fidelities))]))
        else:
            self.schema = schema

        if len(self.schema) != len(self.fidelities):
            raise ValueError('Cost schema does not match number of fidelity levels')

        ### ARCHIVE
        self.archive = self.init_archive(archive)

        ### HIERARCHICAL (AND DIRECT) MODELS
        self.models = {}
        self.direct_models = {}
        for fid_low, fid_high in stagger(reversed(self.fidelities), offsets=(-1, 0)):

            if fid_low is None:
                model = Surrogate.fromname('Kriging', self.archive, fidelity=fid_high, normalized=normalized, kernel=kernel)
                self.direct_models[fid_high] = model
            else:
                model = HierarchicalSurrogate('Kriging', lower_fidelity_model=self.models[fid_low],
                                              candidate_archive=self.archive, fidelities=[fid_high, fid_low],
                                              normalized=normalized, scaling=scaling, kernel=kernel)
                self.direct_models[fid_high] = Surrogate.fromname('Kriging', self.archive, fidelity=fid_high,
                                                                  normalized=normalized, kernel=kernel)
            self.models[fid_high] = model

        self.top_level_model = self.models[self.fidelities[0]]
        self.top_level_model.retrain()
        for fid in self.fidelities:
            self.direct_models[fid].retrain()


        ### ACQUISITION FUNCTION
        self.utility = bo.util.UtilityFunction(kind='ei', kappa=2.576, xi=1.0).utility
        self.acq_max = partial(
            bo.util.acq_max,
            ac=self.utility, gp=self.top_level_model, bounds=self.bounds.T,
            n_warmup=1000, n_iter=50, random_state=np.random.RandomState()
        )

        ### PLOTTING
        self.functions_to_plot = [
            *self.func.functions,
            *[partial(gpplot, func=self.models[fid].predict) for fid in self.fidelities],
            *[partial(gpplot, func=self.models[fid].predict, return_std=True) for fid in self.fidelities],
            *[lambda x: self.utility(x, gp=self.models[fid], y_max=self.archive.max[fid]) for fid in self.fidelities],
        ]
        self.titles = [
            *self.fidelities,
            *[f'{fid} model' for fid in self.fidelities],
            *[f'{fid} std' for fid in self.fidelities],
            *[f'{fid} acq' for fid in self.fidelities],
        ]

        ############ MSE SETUP
        score_fidelities = [f'{fid}_hier' for fid in self.fidelities[:-1]] + \
                         self.fidelities
        self.MSECollection = namedtuple('MSECollection', score_fidelities)
        self.R2Collection = namedtuple('R2Collection', score_fidelities)
        self.MSERecord = namedtuple('MSERecord', ['repetition', 'iteration',
                                                  *(f'mse_{mse}' for mse in score_fidelities)])

        if test_sample is None:
            test_sample = low_lhs_sample(self.ndim, 500 * self.ndim)

        test_sample = rescale(test_sample, range_in=(0,1), range_out=self.input_range)
        self.test_sample = test_sample

        self.mse_tester = {
            fid: partial(mean_squared_error,
                         y_pred=getattr(self.func, fid)(self.test_sample))
            for fid in self.fidelities
        }

        self.r2_tester = {
            fid: partial(r2_score,
                         y_pred=getattr(self.func, fid)(self.test_sample))
            for fid in self.fidelities
        }



    def init_archive(self, archive):
        if archive is not None:
            return archive

        archive_fidelities = self.fidelities + [f'{a}-{b}' for a, b in pairwise(self.fidelities)]
        archive = CandidateArchive(ndim=self.ndim, fidelities=archive_fidelities)

        samples = create_random_sample_set(self.ndim, zip(self.fidelities, [5, 8, 13]),
                                           desired_range=self.input_range)
        for fidelity in self.fidelities:
            archive.addcandidates(
                samples[fidelity],
                getattr(self.func, fidelity)(samples[fidelity]),
                fidelity=fidelity
            )

        return archive



    def set_plotoptions(self):
        pass


    def retrain(self):
        pass


    def run(self, *, num_iters=100, repetition_idx=0):

        records = []

        for iteration_idx in range(num_iters):
            self.iteration(iteration_idx)
            record = self.MSERecord(repetition_idx, iteration_idx, *self.getMSE())
            records.append(record)

        return pd.DataFrame(records)


    def iteration(self, iteration_idx):
        next_points = {fid: None for fid in self.fidelities}

        fid_pairs = stagger(reversed(self.fidelities), offsets=(-1, 0))

        for cost, (fid_low, fid_high) in zip(reversed(self.schema), fid_pairs):
            if iteration_idx % cost == 0:
                next_point = self.limited_acq_max(fid_low=fid_low, fid_high=fid_high)
                next_value = self.func[fid_high](next_point.reshape((-1,1)))
                self.archive.addcandidate(next_point, next_value, fidelity=fid_high)
                self.top_level_model.retrain()
                next_points[fid_high] = next_point

        for fid in self.fidelities:
            self.direct_models[fid].retrain()

        if self.show_plot or self.save_plot:
            self.plot()



    def limited_acq_max(self, fid_low, fid_high):
        if fid_low is None:
            return self.acq_max(y_max=self.archive.max['high'])

        candidates_low = self.archive.getcandidates(fidelity=fid_low).candidates
        candidates_high = self.archive.getcandidates(fidelity=fid_high).candidates
        interesting_candidates = list({tuple(x.tolist()) for x in candidates_low}     # Loving the type transformations here....
                                      - {tuple(y.tolist()) for y in candidates_high})
        predicted_values = self.top_level_model.predict(np.array(interesting_candidates))

        return list(interesting_candidates[np.argmax(predicted_values)])



    def getMSE(self):
        return self.MSECollection(
            *[self.mse_tester[fid](self.models[fid].predict(self.test_sample))
              for fid in self.fidelities[:-1]],
            *[self.mse_tester[fid](self.direct_models[fid].predict(self.test_sample))
              for fid in self.fidelities],
        )



    def getR2(self):
        return self.R2Collection(
            *[self.r2_tester[fid](self.models[fid].predict(self.test_sample))
              for fid in self.fidelities[:-1]],
            *[self.r2_tester[fid](self.direct_models[fid].predict(self.test_sample))
              for fid in self.fidelities],
        )


    def plot(self):

        red_dot = {'marker': '.', 'color': 'red'}
        blue_circle = {'marker': 'o', 'facecolors': 'none', 'color': 'blue'}
        green_cross = {'marker': '+', 'color': 'green'}
        surfaces = createsurfaces(self.functions_to_plot, l_bound=self.func.l_bound, u_bound=self.func.u_bound)
        points = [
                     [ScatterPoints(*self.archive.getcandidates(fidelity='high'), style=red_dot)],
                     [ScatterPoints(*self.archive.getcandidates(fidelity='medium'), style=blue_circle)],
                     [ScatterPoints(*self.archive.getcandidates(fidelity='low'), style=green_cross)],
                 ] * 4

        plotsurfaces(surfaces, all_points=points, titles=self.titles, as_3d=False, shape=(4, 3), show=self.show_plot)
