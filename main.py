#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Class file for a multi-fidelity Bayesian Optimizer with built-in hierarchical
surrogate model
"""

from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
from collections import namedtuple
from operator import itemgetter
from scipy.optimize import minimize
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression

from multiLevelCoSurrogates.multiFidBO import MultiFidelityBO
from multiLevelCoSurrogates.CandidateArchive import CandidateArchive
from multiLevelCoSurrogates.Utils import rescale, low_lhs_sample, ValueRange


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


def split_bi_fidelity_doe(DoE, num_high, num_low):
    """Given an existing bi-fidelity Design of Experiments (DoE) `high, low`,
    creates a subselection of given size `num_high, num_low` based on uniform
    selection. The subselection maintains the property that all high-fidelity
    samples are a subset of the low-fidelity samples.

    Raises a `ValueError` if invalid `num_high` or `num_low` are given."""
    high, low = DoE
    if not 1 < num_high < len(high):
        raise ValueError(f"'num_high' must be in the range [2, len(DoE.high) (={len(DoE.high)})], but is {num_high}")
    elif num_low > len(low):
        raise ValueError(f"'num_low' cannot be greater than len(DoE.low) (={len(DoE.low)}), but is {num_low}")
    elif num_low <= num_high:
        raise ValueError(f"'num_low' must be greater than 'num_high', but {num_low} <= {num_high}")

    indices = np.random.permutation(len(high))
    sub_high, leave_out_high = high[indices[:num_high]], high[indices[num_high:]]

    if num_low == len(low):
        sub_low = low
        leave_out_low = []
    else:
        # remove all sub_high from low
        filtered_low = np.array([x for x in low if x not in sub_high])
        # randomly select (num_low - num_high) remaining
        indices = np.random.permutation(len(filtered_low))
        num_low_left = num_low - num_high
        extra_low, leave_out_low = filtered_low[indices[:num_low_left]], \
                                   filtered_low[indices[num_low_left:]]
        # concatenate sub_high with selected sub_low
        sub_low = np.concatenate([sub_high, extra_low], axis=0)

    selected = BiFidelityDoE(sub_high, sub_low)
    left_out = BiFidelityDoE(leave_out_high, leave_out_low)
    return selected, left_out


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


def create_error_grid(archive, num_reps=50, func=None):

    if not func:
        raise NotImplementedError

    highs = archive.getcandidates(fidelity='high')[0]
    lows = archive.getcandidates(fidelity='low')[0]
    DoE = BiFidelityDoE(highs, lows)

    max_num_high = len(highs)
    max_num_low = len(lows)

    error_grid = np.full((max_num_high+1, max_num_low+1, num_reps+1, 3), np.nan)

    instances = [
        (h, l, r)
        for h, l, r in product(range(2, max_num_high, 2),
                               range(3, max_num_low, 2),
                               range(num_reps))
        if h < l
    ]
    for i, (num_high, num_low, rep) in enumerate(instances):

        if i % 1_000 == 0:
            print(f"{i}/{len(instances)}")

        # Create sub-sampled Multi-Fidelity DoE in- and output according to instance specification
        (high_x, low_x), _ = split_bi_fidelity_doe(DoE, num_high, num_low)
        high_y, low_y = func.high(high_x), \
                        func.low(low_x)

        # Create an archive from the MF-function and MF-DoE data
        archive = CandidateArchive.from_multi_fidelity_function(func, ndim=func.ndim)
        archive.addcandidates(low_x, low_y, fidelity='low')
        archive.addcandidates(high_x, high_y, fidelity='high')

        # (Automatically) Create the hierarchical model
        mfbo = MultiFidelityBO(func, archive)

        # Get the results we're interested in from the model for this instance
        error_grid[num_high, num_low, rep] = mfbo.getMSE()

    models = ['high_hier', 'high', 'low']
    return xr.DataArray(
        error_grid,
        dims=['n_high', 'n_low', 'rep', 'model'],
        coords={
            'n_high': range(max_num_high+1),
            'n_low': range(max_num_low+1),
            'rep': range(num_reps+1),
            'model': models,
        },
    )


def simple_multifid_bo(func, budget, cost_ratio, doe_n_high, doe_n_low, num_reps=50):
    if doe_n_high + cost_ratio*doe_n_low >= budget:
        raise ValueError('Budget should not be exhausted after DoE')

    Entry = namedtuple('Entry', 'budget, time_since_high_eval, tau, fidelity')
    entries = []

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
        EG = create_error_grid(archive, num_reps=num_reps, func=func)

        #fit lin-reg for beta_1, beta_2
        reg = fit_lin_reg(EG)
        beta_1, beta_2 = reg.coef_[:2]

        #determine \tau based on beta_1, beta_2 and cost_ratio
        tau = np.ceil(beta_2 / (beta_1*cost_ratio))  #todo confirm with notes

        if tau <= 1:
            warn('Low-fidelity not expected to add information, no need to use multi-fidelity')

        #compare \tau with current count t to select fidelity
        fidelity = 'high' if time_since_high_eval >= tau else 'low'

        entries.append(Entry(budget, time_since_high_eval, tau, fidelity))

        #predict best place to evaluate:
        if fidelity == 'high':
            #best predicted low-fid only datapoint for high-fid (to maintain hierarchical model)
            all_low = {
                tuple(candidate)
                for candidate in archive.getcandidates(fidelity='low').candidates
            }

            all_high = {
                tuple(candidate)
                for candidate in archive.getcandidates(fidelity='high').candidates
            }

            candidates = [np.array(cand).reshape(1, -1) for cand in all_low - all_high]  # only consider candidates that are not yet evaluated in high-fidelity
            candidate_predictions = [
                (cand, mfbo.models['high'].predict(cand.reshape(1, -1)))
                for cand in candidates
            ]

            x = min(candidate_predictions, key=itemgetter(1))[0].ravel()
            time_since_high_eval = 0
            budget -= 1
        else:  # elif fidelity == 'low':
            #simple optimization for low-fid
            x = minimize(
                lambda x: mfbo.models['high'].predict(x.reshape(1, -1)),
                x0=np.random.uniform(func.l_bound, func.u_bound).reshape(1, -1),
                bounds=func.bounds,
            ).x
            time_since_high_eval += 1
            budget -= cost_ratio

        #evaluate best place
        archive.addcandidate(x, func[fidelity](x.reshape(1, -1)), fidelity=fidelity)

        #update model
        mfbo.retrain()

    return mfbo, pd.DataFrame.from_records(entries, columns=Entry._fields), archive


if __name__ == '__main__':
    import mf2
    from pickle import dump
    for func in [
        mf2.branin,
        mf2.currin,
        mf2.himmelblau,
        mf2.six_hump_camelback,
        mf2.park91a,
        mf2.hartmann6,
        mf2.borehole,
        mf2.bohachevsky,
        mf2.booth,
        mf2.park91b,
    ]:

        _, df, archive = simple_multifid_bo(func, 50, 0.2, 10, 25, num_reps=20)
        df.to_csv(f'{func.name}-tracking.csv')
        with open(f'{func.name}-archive.pkl') as f:
            dump(archive.data, f)
