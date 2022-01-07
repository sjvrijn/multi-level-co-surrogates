#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Experiment file for comparing simple versions of multi-fidelity optimizers"""

import argparse
from csv import writer
from functools import partial
from warnings import warn, simplefilter

import bayes_opt as bo
import mf2
import numpy as np
import pandas as pd
import xarray as xr
from collections import namedtuple
from operator import itemgetter
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from pyprojroot import here

from experiments import scale_to_function, create_subsampling_error_grid, mlcs


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


RANDOM_SEED_BASE = 20160501
RAND_SAMPLES_PER_DIM = 10

save_dir = here('files/2020-11-05-simple-mfbo/', warn=False)
save_dir.mkdir(parents=True, exist_ok=True)
plot_dir = here('plots/2020-11-05-simple-mfbo/', warn=False)
plot_dir.mkdir(parents=True, exist_ok=True)

archive_file_template = 'archive_{:03d}.npy'
errorgrid_file_template = 'errorgrid_{:03d}.nc'



Entry = namedtuple('Entry', 'budget time_since_high_eval tau fidelity candidate fitness')


class Optimizer:
    N_RAND_SAMPLES = 100

    def __init__(self, func, budget, cost_ratio, doe_n_high, doe_n_low, fid_selection_method, num_reps=50):
        if doe_n_high + cost_ratio * doe_n_low >= budget:
            raise ValueError('Budget should not be exhausted after DoE')

        np.random.seed(20160501)

        self.func = func
        self.init_budget = budget
        self.cost_ratio = cost_ratio
        self.num_reps = num_reps
        self.fid_selection_method = fid_selection_method
        self.time_since_high_eval = 0
        self.entries = []
        self.archive = make_mf_doe(func, doe_n_high, doe_n_low)
        # subtract mf-DoE from budget
        self.budget = self.init_budget - (doe_n_high + doe_n_low * cost_ratio)

        if fid_selection_method == 'EG':
            self.proto_eg = mlcs.ProtoEG(self.archive, num_reps=num_reps)
            self.proto_eg.subsample_errorgrid()
        else:
            self.proto_eg = None

        self.mfm = mlcs.MultiFidelityModel(fidelities=['high', 'low'], archive=self.archive,
                                           kernel='Matern', scaling='off')

        self.utility = bo.util.UtilityFunction(kind='ei', kappa=2.576, xi=1.0).utility
        self.acq_max = partial(
            bo.util.acq_max,
            ac=self.utility, gp=self.mfm.top_level_model, bounds=self.func.bounds.T,
            n_warmup=1000, n_iter=50, random_state=np.random.RandomState()
        )


    def iterate(self):  # sourcery skip: assign-if-exp
        while self.budget > 0:
            fidelity = self.select_fidelity()

            # predict best place to evaluate:
            if fidelity == 'high':
                x = self.eval_high_fid()
            else:  # elif fidelity == 'low':
                x = self.eval_low_fid()

            # evaluate best place
            y = self.func[fidelity](x.reshape(1, -1))[0]
            self.archive.addcandidate(candidate=x.flatten(), fitness=y, fidelity=fidelity)

            # update model & error grid
            self.mfm.retrain()
            if self.proto_eg and self.budget > 0:  # prevent unnecessary computation
                self.proto_eg.update_errorgrid_with_sample(x, fidelity=fidelity)

            # logging
            # self.entries.append(Entry(self.budget, self.time_since_high_eval, tau, fidelity, x, y))

        return self.mfm, pd.DataFrame.from_records(self.entries, columns=Entry._fields), self.archive


    def eval_high_fid(self):
        # best predicted low-fid only datapoint for high-fid (to maintain hierarchical model)
        candidates = select_high_fid_only_candidates(self.archive)
        candidate_predictions = [
            (cand, self.utility(
                cand.reshape(1, -1),
                gp=self.mfm.top_level_model,
                y_max=self.archive.max['high'])
             )
            for cand in candidates
        ]

        x = min(candidate_predictions, key=itemgetter(1))[0].ravel()
        self.time_since_high_eval = 0
        self.budget -= 1
        return x


    def eval_low_fid(self):
        self.time_since_high_eval += 1
        self.budget -= self.cost_ratio
        return self.acq_max(y_max=self.archive.max['high'])


    def select_fidelity(self):

        if self.fid_selection_method == 'EG':
            tau = calc_tau_from_EG(self.proto_eg.error_grid['mses'], self.cost_ratio)
            # compare \tau with current count t to select fidelity, must be >= 1
            fidelity = 'high' if 1 <= tau <= self.time_since_high_eval else 'low'
        elif self.fid_selection_method == 'fixed':
            fidelity = 'high' if 1 <= (1 / self.cost_ratio) <= self.time_since_high_eval else 'low'
        else:
            raise NotImplementedError(f"Fidelity selection method '{self.fid_selection_method}' has no implementation")

        return fidelity


def make_mf_doe(func: mf2.MultiFidelityFunction, doe_n_high: int, doe_n_low: int):
    # make mf-DoE
    high_x, low_x = mlcs.bi_fidelity_doe(func.ndim, doe_n_high, doe_n_low)
    high_x, low_x = scale_to_function(func, [high_x, low_x])
    high_y, low_y = func.high(high_x), func.low(low_x)
    # create archive
    return mlcs.CandidateArchive.from_bi_fid_DoE(high_x, low_x, high_y, low_y)


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
def proto_EG_multifid_bo(func, init_budget, cost_ratio, doe_n_high, doe_n_low, run_save_dir, num_reps=50, seed_offset=None):
    if seed_offset:
        np.random.seed(RANDOM_SEED_BASE + seed_offset)
    start = time()

    if doe_n_high + cost_ratio*doe_n_low >= init_budget:
        raise ValueError('Budget should not be exhausted after DoE')

    mfm_opts = dict(
        kernel='Matern',
        scaling='off',
    )
    logfile = run_save_dir / 'log.csv'
    Entry = namedtuple('Entry', 'iteration budget iter_since_high_eval tau fidelity wall_time nhigh nlow reuse_fraction candidate fitness')
    with open(logfile, 'w') as csvfile:
        logwriter = writer(csvfile, delimiter=';')
        logwriter.writerow(Entry._fields)
    entries = []

    #make mf-DoE
    high_x, low_x = mlcs.bi_fidelity_doe(func.ndim, doe_n_high, doe_n_low)
    high_x, low_x = scale_to_function(func, [high_x, low_x])
    high_y, low_y = func.high(high_x), func.low(low_x)

    #subtract mf-DoE from budget
    budget = init_budget - (doe_n_high + doe_n_low*cost_ratio)

    #create archive
    archive = mlcs.CandidateArchive.from_bi_fid_DoE(high_x, low_x, high_y, low_y)

    proto_eg = mlcs.ProtoEG(archive, num_reps=num_reps, interval=2, mfm_opts=mfm_opts)
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
                random_candidates = scale_to_function(func, np.random.rand(RAND_SAMPLES_PER_DIM * func.ndim, func.ndim))
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

        # logging
        entries.append(Entry(iterations, budget, iter_since_high_eval, tau, fidelity, time()-start, archive.count('high'), archive.count('low'), proto_eg.reuse_fraction, x.flatten(), y))
        with open(logfile, 'a') as csvfile:
            logwriter = writer(csvfile, delimiter=';')
            logwriter.writerow(entries[-1])
        np.save(run_save_dir / archive_file_template.format(iterations), archive)
        proto_eg.error_grid.to_netcdf(run_save_dir / errorgrid_file_template.format(iterations))
        iterations += 1


# @timing
def simple_multifid_bo(func, init_budget, cost_ratio, doe_n_high, doe_n_low, run_save_dir, num_reps=50, seed_offset=None):
    if seed_offset:
        np.random.seed(RANDOM_SEED_BASE + seed_offset)
    start = time()

    if doe_n_high + cost_ratio*doe_n_low >= init_budget:
        raise ValueError('Budget should not be exhausted after DoE')

    mfbo_opts = dict(
        kernel='Matern',
    )

    logfile = run_save_dir / 'log.csv'
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

            while x in archive:  # resample to ensure a new candidate is added to the archive
                # print(f'Existing candidate {x} ...')
                random_candidates = scale_to_function(func, np.random.rand(RAND_SAMPLES_PER_DIM * func.ndim, func.ndim))
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
        np.save(run_save_dir / archive_file_template.format(iterations), archive)
        EG.to_netcdf(run_save_dir / errorgrid_file_template.format(iterations))
        iterations += 1

        #update model
        mfbo.retrain()


@timing
def fixed_ratio_multifid_bo(func, init_budget, cost_ratio, doe_n_high, doe_n_low, run_save_dir, seed_offset=None, **_):
    if seed_offset:
        np.random.seed(RANDOM_SEED_BASE + seed_offset)
    start = time()

    if doe_n_high + cost_ratio*doe_n_low >= init_budget:
        raise ValueError('Budget should not be exhausted after DoE')

    logfile = run_save_dir / 'log.csv'
    Entry = namedtuple('Entry', 'iteration budget iter_since_high_eval tau fidelity wall_time nhigh nlow reuse_fraction candidate fitness')
    with open(logfile, 'w') as csvfile:
        logwriter = writer(csvfile, delimiter=';')
        logwriter.writerow(Entry._fields)
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

            while x in archive:  # resample to ensure a new candidate is added to the archive
                # print(f'Existing candidate {x} ...')
                random_candidates = scale_to_function(func, np.random.rand(RAND_SAMPLES_PER_DIM * func.ndim, func.ndim))
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
        np.save(run_save_dir / archive_file_template.format(iterations), archive)
        iterations += 1

        #update model
        mfbo.retrain()


def class_fixed_ratio_multifid_bo(func, init_budget, cost_ratio, doe_n_high, doe_n_low, run_save_dir, seed_offset=None, **_):
    opt = Optimizer(func, init_budget, cost_ratio, doe_n_high, doe_n_low, fid_selection_method='fixed')
    results = opt.iterate()
    print(opt.archive.data)
    return results


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


def main(args):
    import sklearn
    simplefilter("ignore", category=FutureWarning)
    simplefilter("ignore", category=sklearn.exceptions.ConvergenceWarning)
    simplefilter("ignore", category=TauSmallerThanOneWarning)
    simplefilter("ignore", category=mlcs.LowHighFidSamplesWarning)

    functions = [
        mf2.bohachevsky,         # 0
        mf2.booth,               # 1
        mf2.branin,              # 2
        mf2.currin,              # 3
        mf2.himmelblau,          # 4
        mf2.six_hump_camelback,  # 5

        mf2.park91a,             # 6
        mf2.park91b,             # 7

        mf2.hartmann6,           # 8

        mf2.borehole,            # 9
    ]

    if args.idx:
        functions = [functions[f_idx] for f_idx in args.idx]

    kwargs = dict(
        init_budget=args.budget,
        cost_ratio=0.2,
        doe_n_high=5,
        doe_n_low=10,
        num_reps=args.nreps,
    )

    experiment_functions = {
        'fixed': class_fixed_ratio_multifid_bo,
        'naive': simple_multifid_bo,
        'proto-eg': proto_EG_multifid_bo,
    }

    for func in functions:
        print(func.name)

        for idx in range(args.niters):
            kwargs['seed_offset'] = idx

            for name, experiment_func in experiment_functions.items():
                if args.experiment not in [None, name]:
                    continue
                do_run(func, f'{name}-b{args.budget}-i{idx}', experiment_func, kwargs)


def do_run(benchmark_func, experiment_name, run_func, kwargs):
    print(f'    {experiment_name}...')
    run_save_dir = save_dir / f'{benchmark_func.name}-{experiment_name}'
    run_save_dir.mkdir(parents=True, exist_ok=True)
    run_func(
        func=benchmark_func,
        run_save_dir=run_save_dir,
        **kwargs
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('idx', type=int, nargs='*',
                        help='Experiment indices to run. Default: all')
    parser.add_argument('-e', '--experiment', type=str, nargs='?',
                        help='Experiment function to run. Options: fixed, naive, proto-eg. Default: all')
    parser.add_argument('--nreps', type=int, default=50,
                        help='number of independent repetitions to perform for the error grid')
    parser.add_argument('--niters', type=int, default=5,
                        help='number of independent iterations of the experiment to perform')
    parser.add_argument('--budget', type=int, default=25,
                        help='evaluation budget')
    arguments = parser.parse_args()

    main(arguments)
