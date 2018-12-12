#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Class file for a multi-fidelity Bayesian Optimizer
"""

import numpy as np
import pandas as pd
import bayes_opt as bo

from functools import partial
from collections import namedtuple
from sklearn.metrics import mean_squared_error
from pyDOE import lhs
from more_itertools import pairwise

from multiLevelCoSurrogates.bo import gpplot, ScatterPoints
from multiLevelCoSurrogates.CandidateArchive import CandidateArchive
from multiLevelCoSurrogates.Utils import select_subsample, linearscaletransform, \
    sample_by_function, createsurfaces, plotsurfaces, ValueRange
from multiLevelCoSurrogates.Surrogates import Kriging, HierarchicalSurrogate



def create_sample_set(ndim, size_per_fidelity, desired_range=None):
    size_per_fidelity = iter(sorted(size_per_fidelity, key=lambda x: x[1], reverse=True))
    range_lhs = ValueRange(0, 1)

    fid, size = next(size_per_fidelity)
    sample = lhs(n=ndim, samples=size)
    if desired_range is not None:
        sample = linearscaletransform(sample, range_in=range_lhs, range_out=desired_range)
    samples = {fid: sample}
    for fid, size in size_per_fidelity:
        sample = select_subsample(sample.T, num=size).T
        samples[fid] = sample

    return samples



MSECollection = namedtuple('MSECollection', ['high_hier', 'high', 'medium_hier', 'medium', 'low'])
MSERecord = namedtuple('MSERecord', ['repetition', 'iteration',
                                     *('mse_' + mse for mse in MSECollection._fields)])



class MultiFidelityBO:

    def __init__(self, multi_fid_func, archive=None, do_plot=False):

        self.do_plot = do_plot
        self.func = multi_fid_func
        self.ndim = self.func.ndim
        self.bounds = np.array([self.func.l_bound, self.func.u_bound], dtype=np.float)
        self.input_range = ValueRange(*self.bounds)
        self.output_range = ValueRange(-450, 0)
        self.fidelities = list(self.func.fidelity_names)

        self.schema = [4,2,1]
        self.utility = bo.helpers.UtilityFunction(kind='ei', kappa=2.576, xi=1.0).utility

        ### ARCHIVE
        self.archive = self.init_archive(archive)





        ################ MODELS
        self.low_model = Kriging(self.archive, num_points=None, fidelity='low')
        self.medium_hier_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=self.low_model,
                                                       candidate_archive=self.archive, fidelities=self.fidelities[1:3])
        self.high_hier_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=self.medium_hier_model,
                                                     candidate_archive=self.archive, fidelities=self.fidelities[0:2])
        self.high_hier_model.retrain()


        self.medium_model = Kriging(self.archive, num_points=None, fidelity='medium')
        self.medium_model.retrain()
        self.high_model = Kriging(self.archive, num_points=None, fidelity='high')
        self.high_model.retrain()




        self.acq_max = partial(
            bo.helpers.acq_max,
            ac=self.utility, gp=self.high_hier_model, bounds=self.bounds.T,
            n_warmup=1000, n_iter=50, random_state=np.random.RandomState()
        )






        ######### PLOTTING
        self.functions_to_plot = [
            self.func.high, self.func.medium, self.func.low,

            partial(gpplot, func=self.high_hier_model.predict),
            partial(gpplot, func=self.medium_hier_model.predict),
            partial(gpplot, func=self.low_model.predict),

            partial(gpplot, func=self.high_hier_model.predict, return_std=True),
            partial(gpplot, func=self.medium_hier_model.predict, return_std=True),
            partial(gpplot, func=self.low_model.predict, return_std=True),

            lambda x: self.utility(x, gp=self.high_hier_model, y_max=self.archive.max['high']),
            lambda x: self.utility(x, gp=self.medium_hier_model, y_max=self.archive.max['medium']),
            lambda x: self.utility(x, gp=self.low_model, y_max=self.archive.max['low']),
        ]
        self.titles = [
            'high', 'medium', 'low',
            'high model', 'medium model', 'low model',
            'high std', 'medium std', 'low std',
            'acq_high', 'acq_medium', 'acq_low',
        ]



        ############ MSE SETUP
        n_samples = 1000
        self.test_sample = sample_by_function(self.func.high, n_samples=n_samples, ndim=self.ndim,
                                              range_in=self.input_range, range_out=self.output_range)

        self.mse_tester = {
            fid: partial(mean_squared_error,
                         y_pred=getattr(self.func, fid)(self.test_sample))
            for fid in self.fidelities
        }



    def init_archive(self, archive):
        if archive is not None:
            return archive

        archive_fidelities = self.fidelities + [f'{a}-{b}' for a, b in pairwise(self.fidelities)]
        archive = CandidateArchive(ndim=self.ndim, fidelities=archive_fidelities)

        samples = create_sample_set(self.ndim, zip(self.fidelities, [5, 8, 13]),
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
            records.append(self.iteration(iteration_idx, repetition_idx))

        return pd.DataFrame(records)


    def iteration(self, iteration_idx, repetition_idx):
        next_points = [None, None, None]



        if iteration_idx % self.schema[2] == 0:

            next_point = self.limited_acq_max(fid_low=None, fid_high='low')
            next_value = self.func.low(next_point)
            self.archive.addcandidate(next_point, next_value, fidelity='low')
            self.high_hier_model.retrain()

            next_points[2] = next_point

        if iteration_idx % self.schema[1] == 0:

            next_point = self.limited_acq_max(fid_low='low', fid_high='medium')
            next_value = self.func.medium(next_point)
            self.archive.addcandidate(next_point, next_value, fidelity='medium')
            self.high_hier_model.retrain()

            next_points[1] = next_point

        if iteration_idx % self.schema[0] == 0:

            next_point = self.limited_acq_max(fid_low='medium', fid_high='high')
            next_value = self.func.medium(next_point)
            self.archive.addcandidate(next_point, next_value, fidelity='high')
            self.high_hier_model.retrain()

            next_points[0] = next_point

        print(f'iteration: {iteration_idx} | archive_size: {len(self.archive)} | '
              f'next point: {next_points[2]} {next_points[1]} {next_points[0]}')
        self.medium_model.retrain()
        self.high_model.retrain()

        if self.do_plot:
            self.plot()

        mses = self.getMSE()
        record = MSERecord(repetition_idx, iteration_idx, *mses)

        return record



    def limited_acq_max(self, fid_low, fid_high):
        if fid_low is None:
            return self.acq_max(y_max=self.archive.max['high'])

        candidates_low = self.archive.getcandidates(fidelity=fid_low).candidates
        candidates_high = self.archive.getcandidates(fidelity=fid_high).candidates
        interesting_candidates = list({tuple(x.tolist()) for x in candidates_low}
                                      - {tuple(y.tolist()) for y in candidates_high})
        predicted_values = self.high_hier_model.predict(np.array(interesting_candidates))

        return list(interesting_candidates[np.argmax(predicted_values)])



    def getMSE(self):
        return MSECollection(
            self.mse_tester['high'](self.high_hier_model.predict(self.test_sample)),
            self.mse_tester['high'](self.high_model.predict(self.test_sample)),
            self.mse_tester['medium'](self.medium_hier_model.predict(self.test_sample)),
            self.mse_tester['medium'](self.medium_model.predict(self.test_sample)),
            self.mse_tester['low'](self.low_model.predict(self.test_sample)),
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

        plotsurfaces(surfaces, all_points=points, titles=self.titles, as_3d=False, shape=(4, 3))
