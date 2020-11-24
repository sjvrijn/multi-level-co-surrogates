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
from collections import namedtuple
from operator import itemgetter
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from pyprojroot import here

from experiments import bi_fidelity_doe, scale_to_function, create_subsampling_error_grid, mlcs


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


def simple_multifid_bo(func, budget, cost_ratio, doe_n_high, doe_n_low, num_reps=50):
    np.random.seed(20160501)

    if doe_n_high + cost_ratio*doe_n_low >= budget:
        raise ValueError('Budget should not be exhausted after DoE')

    Entry = namedtuple('Entry', 'budget time_since_high_eval tau fidelity candidate fitness')
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
        tau = np.ceil(1 / (beta_1 / (beta_2 / cost_ratio)))

        if tau <= 1:
            warn('Low-fidelity not expected to add information, no need to use multi-fidelity')

        #compare \tau with current count t to select fidelity, must be >= 1
        fidelity = 'high' if 1 <= tau <= time_since_high_eval else 'low'

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
                bounds=func.bounds.T,
            ).x
            time_since_high_eval += 1
            budget -= cost_ratio

        #evaluate best place
        y = func[fidelity](x.reshape(1, -1))
        archive.addcandidate(x, y, fidelity=fidelity)
        entries.append(Entry(budget, time_since_high_eval, tau, fidelity, x, y))

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