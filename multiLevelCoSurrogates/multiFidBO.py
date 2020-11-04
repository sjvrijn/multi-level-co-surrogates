#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Class file for a multi-fidelity Bayesian Optimizer with built-in hierarchical
surrogate model
"""

import numpy as np
import pandas as pd
import xarray as xr
import bayes_opt as bo

from functools import partial
from collections import namedtuple
from operator import itemgetter
from scipy.optimize import minimize
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from more_itertools import pairwise, stagger

from .CandidateArchive import CandidateArchive
from .Utils import create_random_sample_set, rescale, \
    low_lhs_sample, createsurfaces, plotsurfaces, gpplot, ScatterPoints, ValueRange
from .Surrogates import HierarchicalSurrogate, Surrogate


BiFidelityDoE = namedtuple("BiFidelityDoE", "high low")


def bi_fidelity_doe(ndim, num_high, num_low):
    """Create a Design of Experiments (DoE) for two fidelities in `ndim`
    dimensions. The high-fidelity samples are guaranteed to be a subset
    of the low-fidelity samples.

    :returns high-fidelity samples, low-fidelity samples
    """
    high_x = low_lhs_sample(ndim, num_high)
    low_x = low_lhs_sample(ndim, num_low)

    dists = distance.cdist(high_x, low_x)

    #TODO: this is the naive method, potentially speed up?
    highs_to_match = set(range(num_high))
    while highs_to_match:
        min_dist = np.min(dists)
        high_idx, low_idx = np.argwhere(dists == min_dist)[0]

        low_x[low_idx] = high_x[high_idx]
        # make sure just selected samples are not re-selectable
        dists[high_idx,:] = np.inf
        dists[:,low_idx] = np.inf
        highs_to_match.remove(high_idx)
    return BiFidelityDoE(high_x, low_x)


def scale_to_function(func, xx, range_in=ValueRange(0, 1)):
    range_out = (np.array(func.l_bound), np.array(func.u_bound))
    return [rescale(x, range_in=range_in, range_out=range_out) for x in xx]


def fit_lin_reg(da: xr.DataArray, calc_SSE: bool=False):
    """Return lin-reg coefficients after training index -> value"""

    series = da.to_series().dropna()
    X = np.array(series.index.tolist())[:,:2]  # remove rep_idx (3rd column)
    y = np.log10(series.values)
    reg = LinearRegression().fit(X, y)

    if not calc_SSE:
        return reg

    pred_y = reg.predict(X)
    SSE = np.sum((pred_y - y)**2)
    return reg, SSE


def create_error_grid(archive, num_reps=50):



    pass


def simple_multifid_bo(func, budget, cost_ratio, doe_n_high, doe_n_low):
    if doe_n_high + cost_ratio*doe_n_low >= budget:
        raise ValueError('Budget should not be exhausted after DoE')

    #make mf-DoE
    high_x, low_x = bi_fidelity_doe(func.ndim, doe_n_high, doe_n_low)
    high_x, low_x = scale_to_function(func, [high_x, low_x])
    high_y, low_y = func.high(high_x), \
                    func.low(low_x)

    #subtract mf-DoE from budget
    budget -= doe_n_high
    budget -= doe_n_low * cost_ratio

    #create archive
    archive = CandidateArchive.from_multi_fidelity_function(func, ndim=func.ndim)
    archive.addcandidates(low_x, low_y, fidelity='low')
    archive.addcandidates(high_x, high_y, fidelity='high')

    #make mf-model using archive
    mfbo = MultiFidelityBO(func, archive)

    time_since_high_eval = 0
    while budget > 0:
        #select next fidelity to evaluate:
        #sample error grid
        EG = create_error_grid(archive, num_reps=50)
        #fit lin-reg for beta_1, beta_2
        reg = fit_lin_reg(EG)
        beta_1, beta_2 = reg.coef_[:2]
        #determine \tau based on beta_1, beta_2 and cost_ratio
        tau = np.ceil(beta_2 / (beta_1*cost_ratio))  #todo confirm with notes
        #compare \tau with current count t to select fidelity
        fidelity = 'high' if time_since_high_eval > tau else fidelity = 'low'


        #predict best place to evaluate:
        if fidelity == 'high':
            #best predicted low-fid only datapoint for high-fid (to maintain hierarchical model)
            all_low = set(tuple(candidate) for candidate, fitness in archive.getcandidates(fidelity='low'))
            all_high = set(tuple(candidate) for candidate, fitness in archive.getcandidates(fidelity='high'))

            candidates = [np.array(cand).reshape(-1, 1) for cand in all_low - all_high]  # only consider candidates that are not yet evaluated in high-fidelity
            candidate_predictions = [
                (cand, mfbo.models['high_hier'].predict(cand))
                for cand in candidates
            ]
            x = min(candidate_predictions, key=itemgetter(1))[0]
        else:  #elif fidelity == 'low':
            #simple optimization for low-fid
            x = minimize(
                lambda x: mfbo.models['high_hier'].predict(x.reshape(1, -1))[0],
                x0=np.random.uniform(func.l_bound, func.u_bound).reshape(1, -1),
                bounds=func.bounds,
            )

        #evaluate best place
        archive.addcandidate(x, func[fidelity](x), fidelity=fidelity)

        #update model
        mfbo.retrain()
    pass



    # \Require{Budget $b$, Cost ratio $\costratio$, Initial DoE size $\nhigh, \nlow$}
    # \State{$\mathcal{A} \leftarrow $ DoE($\nhigh, \nlow$)} \Comment{Archive of evaluated solutions}
    # \State{$b \leftarrow b - (\nhigh + \nlow\costratio)$}
    # \State{$t \leftarrow 0$} \Comment{time since last high-fidelity evaluation}
    # \While{remaining budget $b > 0$}
    #     \State{Create Error Grid by subsampling from $\mathcal{A}$}
    #     \State{Measure angle $\theta$} \Comment{In practice, the ratio $\frac{\beta_1}{\beta_2}$ is enough}
    #     \State{determine ideal ratio $1/\tau$} \Comment{No MF-utility if $\tau < 1$!}
    #     \State{$M \leftarrow $ train multi-fidelity model on $\mathcal{A}$}
    #     \State{Find next sample location $\Vec{x}$ using model $M$}
    #     \If{$t < \tau$}
    #         \State{Evaluate $\Vec{x}$ in \emph{low} fidelity, add to $\mathcal{A}$}
    #         \State{$t \leftarrow t + 1$}
    #         \State{$b \leftarrow b - \costratio$}
    #     \Else
    #         \State{Evaluate $\Vec{x}$ in \emph{high} fidelity, add to $\mathcal{A}$}
    #         \State{$t \leftarrow 0$}
    #         \State{$b \leftarrow b - 1$}
    #     \EndIf
    # \EndWhile
    # \State{\Return{$\mathcal{A}$}} \Comment{or whatever result is desired from optimization}


    pass


class MultiFidelityBO:

    def __init__(self, multi_fid_func, archive=None, save_plot=False, show_plot=False,
                 schema=None, normalized=True, test_sample=None, minimize=False,
                 surrogate_name='Kriging', kernel=None, scaling='on'):

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

        if len(self.func.u_bound) == self.ndim:
            self.bounds = self.func.bounds
        elif len(self.func.u_bound) == 1:
            self.bounds = np.array([np.repeat(self.func.l_bound, self.ndim),
                                    np.repeat(self.func.u_bound, self.ndim)],
                                   dtype=np.float)
        else:
            raise ValueError(f"Unclear how to extend lower/upper "
                             f"bound into {self.ndim} dimensions")

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
                model = Surrogate.fromname(surrogate_name, self.archive, fidelity=fid_high,
                                           normalized=normalized, kernel=kernel)
                self.direct_models[fid_high] = model
            else:
                model = HierarchicalSurrogate(surrogate_name, lower_fidelity_model=self.models[fid_low],
                                              candidate_archive=self.archive, fidelities=[fid_high, fid_low],
                                              normalized=normalized, scaling=scaling, kernel=kernel)
                self.direct_models[fid_high] = Surrogate.fromname(surrogate_name, self.archive,
                                                                  fidelity=fid_high, normalized=normalized,
                                                                  kernel=kernel)
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
                         y_pred=self.func[fid](self.test_sample))
            for fid in self.fidelities
        }

        self.r2_tester = {
            fid: partial(r2_score,
                         y_pred=self.func[fid](self.test_sample))
            for fid in self.fidelities
        }


    def init_archive(self, archive):
        if archive is not None:
            return archive

        archive = CandidateArchive.from_multi_fidelity_function(self.func, ndim=self.ndim)

        samples = create_random_sample_set(self.ndim, zip(self.fidelities, [5, 8, 13]),
                                           desired_range=self.input_range)
        for fidelity in self.fidelities:
            archive.addcandidates(
                samples[fidelity],
                self.func[fidelity](samples[fidelity]),
                fidelity=fidelity
            )

        return archive


    def set_plotoptions(self):
        pass


    def retrain(self):
        raise NotImplementedError


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
