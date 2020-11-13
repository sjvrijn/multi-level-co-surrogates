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
from sklearn.linear_model import LinearRegression
from pyprojroot import here

from experiments import bi_fidelity_doe, scale_to_function, split_bi_fidelity_doe, \
    BiFidelityDoE, mlcs


save_dir = here('files/2020-11-05-simple-mfbo/')
save_dir.mkdir(parents=True, exist_ok=True)

#TODO de-duplicate (already present in processing.py
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


def create_subsampling_error_grid(
        archive: mlcs.CandidateArchive,
        num_reps: int=50,
        interval: int=2,
        func=None
) -> xr.DataArray:
    """Create an error grid through subsampling from the given archive

    :param archive:  `mlcs.CandidateArchive` containing all available evaluated candidates
    :param num_reps: Number of independent repetitions for each size
    :param interval: Interval at which to fill the error grid
    :param func:     Multi-fidelity function to re-evaluate known candidates  #TODO: remove argument
    :return:         `xr.DataArray` ErrorGrid
    """

    if not func:
        # TODO: implement that the original function is not re-evaluated
        # but that values from `archive` are re-used.
        raise NotImplementedError

    highs = archive.getcandidates(fidelity='high').candidates
    lows = archive.getcandidates(fidelity='low').candidates
    DoE = BiFidelityDoE(highs, lows)

    max_num_high = len(highs)
    max_num_low = len(lows)

    error_grid = np.full((max_num_high+1, max_num_low+1, num_reps+1, 3), np.nan)

    instances = [
        (h, l, r)
        for h, l, r in product(range(2, max_num_high, interval),
                               range(3, max_num_low, interval),
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
        archive = mlcs.CandidateArchive.from_multi_fidelity_function(func, ndim=func.ndim)
        archive.addcandidates(low_x, low_y, fidelity='low')
        archive.addcandidates(high_x, high_y, fidelity='high')

        # (Automatically) Create the hierarchical model
        mfbo = mlcs.MultiFidelityBO(func, archive)

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
    archive = mlcs.CandidateArchive.from_multi_fidelity_function(func, ndim=func.ndim)
    archive.addcandidates(low_x, low_y, fidelity='low')
    archive.addcandidates(high_x, high_y, fidelity='high')

    #make mf-model using archive
    mfbo = mlcs.MultiFidelityBO(func, archive)

    time_since_high_eval = 0
    while budget > 0:
        #select next fidelity to evaluate:
        #sample error grid
        EG = create_subsampling_error_grid(archive, num_reps=num_reps, func=func)

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

            selected_candidates = all_low - all_high

            candidates = [np.array(cand).reshape(1, -1) for cand in selected_candidates]  # only consider candidates that are not yet evaluated in high-fidelity
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
                x0=np.random.uniform(func.l_bound, func.u_bound).reshape(-1, ),
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

        _, df, archive = simple_multifid_bo(
            func=func,
            budget=50,
            cost_ratio=0.2,
            doe_n_high=10,
            doe_n_low=25,
            num_reps=20
        )
        df.to_csv(save_dir / f'{func.name}-tracking.csv')
        with open(save_dir / f'{func.name}-archive.pkl', 'wb') as f:
            dump(str(archive.data), f)