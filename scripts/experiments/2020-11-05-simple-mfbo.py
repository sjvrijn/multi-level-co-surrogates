#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Experiment file for comparing simple versions of multi-fidelity optimizers"""

import argparse
from csv import writer
from pickle import dump
from warnings import warn, simplefilter

import mf2
import numpy as np
import pandas as pd
import xarray as xr
from collections import namedtuple
from operator import itemgetter
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from pyprojroot import here

from experiments import scale_to_function, create_subsampling_error_grid, mlcs, plot_archive


from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        start = time()
        result = f(*args, **kw)
        end = time()
        print(f'func {f.__name__}: {end-start:2.4f} sec')
        return result
    return wrap


save_dir = here('files/2020-11-05-simple-mfbo/', warn=False)
save_dir.mkdir(parents=True, exist_ok=True)
plot_dir = here('plots/2020-11-05-simple-mfbo/', warn=False)
plot_dir.mkdir(parents=True, exist_ok=True)

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


# @timing
def proto_EG_multifid_bo(func, init_budget, cost_ratio, doe_n_high, doe_n_low, run_save_dir, num_reps=50):
    #np.random.seed(20160501)
    N_RAND_SAMPLES = 100
    start = time()

    if doe_n_high + cost_ratio*doe_n_low >= init_budget:
        raise ValueError('Budget should not be exhausted after DoE')

    mfm_opts = dict(
        kernel='Matern',
        scaling='off',
    )
    logfile = run_save_dir / f'log.csv'
    archive_file_template = 'archive_{}.pkl'
    errorgrid_file_template = 'errorgrid_{}.nc'

    Entry = namedtuple('Entry', 'iteration budget iter_since_high_eval tau fidelity wall_time nhigh nlow reuse_fraction candidate fitness')
    with open(logfile, 'w') as csvfile:
        logwriter = writer(csvfile, delimiter=';')
        logwriter.writerow(Entry._fields)
    entries = []

    #make mf-DoE
    high_x, low_x = mlcs.bi_fidelity_doe(func.ndim, doe_n_high, doe_n_low)
    high_x, low_x = scale_to_function(func, [high_x, low_x])
    high_y, low_y = func.high(high_x), \
                    func.low(low_x)

    #subtract mf-DoE from budget
    budget = init_budget - (doe_n_high + doe_n_low*cost_ratio)

    #create archive
    archive = mlcs.CandidateArchive.from_bi_fid_DoE(high_x, low_x, high_y, low_y)

    proto_eg = mlcs.ProtoEG(archive, num_reps=num_reps, mfm_opts=mfm_opts)
    proto_eg.subsample_errorgrid()

    mfm = mlcs.MultiFidelityModel(fidelities=['high', 'low'], archive=archive, **mfm_opts)

    iterations = 0
    iter_since_high_eval = 0
    while budget > 0:
        tau = calc_tau_from_EG(proto_eg.error_grid['mses'], cost_ratio)

        fidelity = 'high' if tau <= iter_since_high_eval and archive.count('high') < archive.count('low') else 'low'

        # predict best place to evaluate:
        if fidelity == 'high':
            #best predicted low-fid only datapoint for high-fid (to maintain hierarchical model)
            candidates = select_high_fid_only_candidates(archive)
            candidate_predictions = [
                (cand, mfm.models['high'].predict(cand.reshape(1, -1)))
                for cand in candidates
            ]

            x = min(candidate_predictions, key=itemgetter(1))[0].ravel()
            iter_since_high_eval = 0
            budget -= 1
        else:  # elif fidelity == 'low':
            # simple optimization for low-fid
            x = minimize(
                lambda x: mfm.models['high'].predict(x.reshape(1, -1)),
                x0=np.random.uniform(func.l_bound, func.u_bound).reshape(-1, ),
                bounds=func.bounds.T,
            ).x

            while x in archive:  # resample to ensure a new candidate is added to the archive
                # print(f'Existing candidate {x} ...')
                random_candidates = scale_to_function(func, np.random.rand(N_RAND_SAMPLES, func.ndim))
                fitnesses = mfm.models['high'].predict(random_candidates)
                x = random_candidates[np.argmin(fitnesses)]
                # print(f'... replaced by {x}')

            iter_since_high_eval += 1
            budget -= cost_ratio

        #evaluate best place
        y = func[fidelity](x.reshape(1, -1))[0]
        archive.addcandidate(candidate=x.flatten(), fitness=y, fidelity=fidelity)

        # update model & error grid
        mfm.retrain()
        proto_eg.update_errorgrid_with_sample(x, fidelity=fidelity)
        # plot_title = f'{func.ndim}D {func.name} with {budget:.1f} budget left'
        # proto_eg.plot_errorgrid(
        #     title=plot_title,
        #     as_log=True,
        #     save_as=plot_dir / f'protoeg-EG-opt-{func.name}-{budget/cost_ratio:.0f}',
        #     save_exts=('png',),
        #     xlim=(3, init_budget),
        #     ylim=(2, (init_budget // 2)),
        # )
        # try:
        #     plot_archive(
        #         archive,
        #         func,
        #         title=plot_title,
        #         save_as=plot_dir / f'protoeg-archive-opt-{func.name}-{budget/cost_ratio:.0f}',
        #         save_exts=('png',),
        #     )
        # except NotImplementedError:
        #     pass

        # logging
        entries.append(Entry(iterations, budget, iter_since_high_eval, tau, fidelity, time()-start, archive.count('high'), archive.count('low'), proto_eg.reuse_fraction, x.flatten(), y))
        with open(logfile, 'a') as csvfile:
            logwriter = writer(csvfile, delimiter=';')
            logwriter.writerow(entries[-1])
        with open(run_save_dir / archive_file_template.format(iterations), 'wb') as f:
            dump(str(archive.data), f)
        proto_eg.error_grid.to_netcdf(run_save_dir / errorgrid_file_template.format(iterations))
        iterations += 1

    return mfm, pd.DataFrame.from_records(entries, columns=Entry._fields), archive


# @timing
def simple_multifid_bo(func, init_budget, cost_ratio, doe_n_high, doe_n_low, run_save_dir, num_reps=50):
    #np.random.seed(20160501)
    start = time()

    if doe_n_high + cost_ratio*doe_n_low >= init_budget:
        raise ValueError('Budget should not be exhausted after DoE')

    mfbo_opts = dict(
        kernel='Matern',
    )

    logfile = run_save_dir / f'log.csv'
    archive_file_template = 'archive_{}.pkl'
    errorgrid_file_template = 'errorgrid_{}.nc'

    Entry = namedtuple('Entry', 'iteration budget iter_since_high_eval tau fidelity wall_time nhigh nlow reuse_fraction candidate fitness')
    entries = []

    #make mf-DoE
    high_x, low_x = mlcs.bi_fidelity_doe(func.ndim, doe_n_high, doe_n_low)
    high_x, low_x = scale_to_function(func, [high_x, low_x])
    high_y, low_y = func.high(high_x), \
                    func.low(low_x)

    #subtract mf-DoE from budget
    budget = init_budget - (doe_n_high + doe_n_low*cost_ratio)

    #create archive
    archive = mlcs.CandidateArchive.from_bi_fid_DoE(high_x, low_x, high_y, low_y)

    #make mf-model using archive
    mfbo = mlcs.MultiFidelityBO(func, archive, **mfbo_opts)

    iterations = 0
    iter_since_high_eval = 0
    while budget > 0:
        #select next fidelity to evaluate:
        #sample error grid
        EG = create_subsampling_error_grid(archive, num_reps=num_reps, func=func, **mfbo_opts)
        tau = calc_tau_from_EG(EG, cost_ratio)

        fidelity = 'high' if tau <= iter_since_high_eval and archive.count('high') < archive.count('low') else 'low'

        #predict best place to evaluate:
        if fidelity == 'high':
            #best predicted low-fid only datapoint for high-fid (to maintain hierarchical model)
            candidates = select_high_fid_only_candidates(archive)
            candidate_predictions = [
                (cand, mfbo.models['high'].predict(cand.reshape(1, -1)))
                for cand in candidates
            ]

            x = min(candidate_predictions, key=itemgetter(1))[0].ravel()
            iter_since_high_eval = 0
            budget -= 1
        else:  # elif fidelity == 'low':
            #simple optimization for low-fid
            x = minimize(
                lambda x: mfbo.models['high'].predict(x.reshape(1, -1)),
                x0=np.random.uniform(func.l_bound, func.u_bound).reshape(-1, ),
                bounds=func.bounds.T,
            ).x

            N_RAND_SAMPLES = 100
            while x in archive:  # resample to ensure a new candidate is added to the archive
                # print(f'Existing candidate {x} ...')
                random_candidates = scale_to_function(func, np.random.rand(N_RAND_SAMPLES, func.ndim))
                fitnesses = mfbo.models['high'].predict(random_candidates)
                x = random_candidates[np.argmin(fitnesses)]
                # print(f'... replaced by {x}')

            iter_since_high_eval += 1
            budget -= cost_ratio

        #evaluate best place
        y = func[fidelity](x.reshape(1, -1))[0]
        archive.addcandidate(x, y, fidelity=fidelity)
        entries.append(Entry(iterations, budget, iter_since_high_eval, tau, fidelity, time()-start, archive.count('high'), archive.count('low'), 0, x.flatten(), y))
        with open(logfile, 'a') as csvfile:
            logwriter = writer(csvfile, delimiter=';')
            logwriter.writerow(entries[-1])
        with open(run_save_dir / archive_file_template.format(iterations), 'wb') as f:
            dump(str(archive.data), f)
        EG.to_netcdf(run_save_dir / errorgrid_file_template.format(iterations))
        iterations += 1

        #update model
        mfbo.retrain()

    return mfbo, pd.DataFrame.from_records(entries, columns=Entry._fields), archive


@timing
def fixed_ratio_multifid_bo(func, init_budget, cost_ratio, doe_n_high, doe_n_low, run_save_dir, num_reps=50):
    np.random.seed(20160501)
    start = time()

    if doe_n_high + cost_ratio*doe_n_low >= init_budget:
        raise ValueError('Budget should not be exhausted after DoE')

    logfile = run_save_dir / f'log.csv'
    archive_file_template = 'archive_{}.pkl'
    errorgrid_file_template = 'errorgrid_{}.nc'

    Entry = namedtuple('Entry', 'iteration budget iter_since_high_eval tau fidelity wall_time nhigh nlow reuse_fraction candidate fitness')
    entries = []

    tau = 1 / cost_ratio

    #make mf-DoE
    high_x, low_x = mlcs.bi_fidelity_doe(func.ndim, doe_n_high, doe_n_low)
    high_x, low_x = scale_to_function(func, [high_x, low_x])
    high_y, low_y = func.high(high_x), \
                    func.low(low_x)

    #subtract mf-DoE from budget
    budget = init_budget - (doe_n_high + doe_n_low*cost_ratio)

    #create archive
    archive = mlcs.CandidateArchive.from_multi_fidelity_function(func, ndim=func.ndim)
    archive.addcandidates(low_x, low_y, fidelity='low')
    archive.addcandidates(high_x, high_y, fidelity='high')

    #make mf-model using archive
    mfbo = mlcs.MultiFidelityBO(func, archive)

    iterations = 0
    iter_since_high_eval = 0
    while budget > 0:
        #select next fidelity to evaluate:
        fidelity = 'high' if tau <= iter_since_high_eval and archive.count('high') < archive.count('low') else 'low'

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
            iter_since_high_eval = 0
            budget -= 1
        else:  # elif fidelity == 'low':
            #simple optimization for low-fid
            x = minimize(
                lambda x: mfbo.models['high'].predict(x.reshape(1, -1)),
                x0=np.random.uniform(func.l_bound, func.u_bound).reshape(-1, ),
                bounds=func.bounds.T,
            ).x

            N_RAND_SAMPLES = 100
            while x in archive:  # resample to ensure a new candidate is added to the archive
                # print(f'Existing candidate {x} ...')
                random_candidates = scale_to_function(func, np.random.rand(N_RAND_SAMPLES, func.ndim))
                fitnesses = mfbo.models['high'].predict(random_candidates)
                x = random_candidates[np.argmin(fitnesses)]
                # print(f'... replaced by {x}')

            iter_since_high_eval += 1
            budget -= cost_ratio

        #evaluate best place
        y = func[fidelity](x.reshape(1, -1))[0]
        archive.addcandidate(x, y, fidelity=fidelity)
        entries.append(Entry(iterations, budget, iter_since_high_eval, tau, fidelity, time()-start, archive.count('high'), archive.count('low'), 0, x.flatten(), y))
        with open(logfile, 'a') as csvfile:
            logwriter = writer(csvfile, delimiter=';')
            logwriter.writerow(entries[-1])
        with open(run_save_dir / archive_file_template.format(iterations), 'wb') as f:
            dump(str(archive.data), f)
        iterations += 1

        #update model
        mfbo.retrain()

    return mfbo, pd.DataFrame.from_records(entries, columns=Entry._fields), archive


def select_high_fid_only_candidates(archive):
    all_low = {
        tuple(candidate)
        for candidate in archive.getcandidates(fidelity='low').candidates
    }
    all_high = {
        tuple(candidate)
        for candidate in archive.getcandidates(fidelity='high').candidates
    }
    selected_candidates = all_low - all_high
    return [np.array(cand).reshape(1, -1) for cand in selected_candidates]


class TauSmallerThanOneWarning(UserWarning):
    """warns that fidelity-selection parameter 'tau' is smaller than one"""

def calc_tau_from_EG(EG, cost_ratio):
    # fit lin-reg for beta_1, beta_2
    reg = fit_lin_reg(EG)
    beta_1, beta_2 = reg.coef_[:2]
    # determine \tau based on beta_1, beta_2 and cost_ratio
    tau = np.ceil(1 / (beta_1 / (beta_2 / cost_ratio)))
    if tau <= 1:
        warn('Low-fidelity not expected to add information', category=TauSmallerThanOneWarning)
    return tau


def main(idx=None):
    import sklearn
    simplefilter("ignore", category=FutureWarning)
    simplefilter("ignore", category=sklearn.exceptions.ConvergenceWarning)
    simplefilter("ignore", category=TauSmallerThanOneWarning)
    simplefilter("ignore", category=mlcs.LowHighFidSamplesWarning)
    num_iters = 5
    np.random.seed(20160501)

    functions = [
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
    ]

    if idx is not None:
        functions = [functions[idx]]

    for func in functions:
        print(func.name)
        for budget in [25]:  # 8, 9, 10, 12, 14, 16, 18, 20, 25, 30]:

            kwargs = dict(
                init_budget=budget,
                cost_ratio=0.2,
                doe_n_high=5,
                doe_n_low=10,
                num_reps=2,
            )
            for idx in range(num_iters):
                do_run(func, f'fixed-b{budget}-i{idx}', fixed_ratio_multifid_bo, kwargs)
                do_run(func, f'naive-b{budget}-i{idx}', simple_multifid_bo, kwargs)
                do_run(func, f'proto-eg-b{budget}-i{idx}', proto_EG_multifid_bo, kwargs)


def do_run(func, name, run_func, kwargs):
    print(f'    {name}...')
    run_save_dir = save_dir / name
    run_save_dir.mkdir(parents=True, exist_ok=True)
    _, df, archive = run_func(
        func=func,
        run_save_dir=run_save_dir,
        **kwargs
    )
    # df.to_csv(save_dir / f'{func.name}-tracking-{name}.csv')
    # with open(save_dir / f'{func.name}-archive-{name}.pkl', 'wb') as f:
    #     dump(str(archive.data), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idx', type=int, nargs='?')
    args = parser.parse_args()

    main(args.idx)
