#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Experiment file for comparing simple versions of multi-fidelity optimizers"""

import argparse
from csv import writer
from functools import partial
from itertools import product
from pathlib import Path
from pprint import pprint
from time import time
from warnings import warn, simplefilter, catch_warnings

import mf2
import numpy as np
import pandas as pd
import xarray as xr
from collections import namedtuple
from operator import itemgetter
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from pyprojroot import here

from experiments import scale_to_function, create_subsampling_error_grid, mlcs


RANDOM_SEED_BASE = 20160501
RAND_SAMPLES_PER_DIM = 10

save_dir = here('files/2020-11-05-simple-mfbo/', warn=False)
save_dir.mkdir(parents=True, exist_ok=True)
plot_dir = here('plots/2020-11-05-simple-mfbo/', warn=False)
plot_dir.mkdir(parents=True, exist_ok=True)

FOLDER_NAME_TEMPLATE = '{func_name}-{name}-c{cost_ratio:f}-b{budget:d}-i{idx:d}'
archive_file_template = 'archive_{:03d}.npy'
errorgrid_file_template = 'errorgrid_{:03d}.nc'


Entry = namedtuple('Entry', 'iteration budget time_since_high_eval tau fidelity wall_time nhigh nlow reuse_fraction candidate fitness')


class UtilityFunction:
    """
    Code adapted from:
      https://github.com/fmfn/BayesianOptimization/blob/380b0d52ae0e3650b023c4ef6db43f7343c75dea/bayes_opt/util.py
    Under MIT License

    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa=2.576, xi=1, kappa_decay=1, kappa_decay_delay=0):

        self.kappa = kappa
        self.xi = xi

        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay
        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'ei_orig', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_best, goal='maximize'):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa, goal)
        if self.kind == 'ei':
            return self._ei(x, gp, y_best, self.xi, goal)
        if self.kind == 'ei_orig':
            return self._ei_orig(x, gp, y_best, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_best, self.xi, goal)

    @staticmethod
    def _ucb(x, gp, kappa, goal):
        with catch_warnings():
            simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
        if goal == 'maximize':
            return mean + kappa * std
        elif goal == 'minimize':
            return mean - kappa * std

    @staticmethod
    def _ei(x, gp, y_best, xi, goal):
        with catch_warnings():
            simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        mean, std = mean.reshape(-1, 1), std.reshape(-1, 1)

        if goal == 'maximize':
            a = (mean - y_best - xi)
        elif goal == 'minimize':
            a = (y_best - mean + xi)

        z = a / std

        if goal == 'maximize':
            return a * norm.cdf(z) + std * norm.pdf(z)
        elif goal == 'minimize':
            return a * norm.cdf(z) - std * norm.pdf(z)

    @staticmethod
    def _ei_orig(x, gp, y_max, xi):
        with catch_warnings():
            simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_best, xi, goal):
        with catch_warnings():
            simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
        if goal == 'maximize':
            z = (mean - y_best - xi) / std
        elif goal == 'minimize':
            z = (y_best - mean - xi) / std
        return norm.cdf(z)


class Optimizer:
    N_RAND_SAMPLES = 100

    def __init__(
        self,
        func,
        budget: int,
        cost_ratio: float,
        doe_n_high: int,
        doe_n_low: int,
        run_save_dir: Path,
        fid_selection_method: str,
        seed_offset: int=0,
        num_reps: int=50,
        goal: str='minimize',
    ):

        if doe_n_high + cost_ratio * doe_n_low >= budget:
            raise ValueError('Budget should not be exhausted after DoE')

        np.random.seed(RANDOM_SEED_BASE + seed_offset)

        self.func = func
        self.init_budget = budget
        self.cost_ratio = cost_ratio
        self.run_save_dir = run_save_dir
        self.fid_selection_method = fid_selection_method
        self.num_reps = num_reps
        self.goal = goal
        self.time_since_high_eval = 0
        self.entries = []
        self.archive = make_mf_doe(func, doe_n_high, doe_n_low)
        # subtract mf-DoE from budget
        self.budget = self.init_budget - (doe_n_high + doe_n_low * cost_ratio)

        self.logfile = run_save_dir / 'log.csv'
        with open(self.logfile, 'w') as csvfile:
            logwriter = writer(csvfile, delimiter=';')
            logwriter.writerow(Entry._fields)

        if fid_selection_method == 'EG':
            self.proto_eg = mlcs.ProtoEG(self.archive, num_reps=num_reps)
            self.proto_eg.subsample_errorgrid()
        else:
            self.proto_eg = None

        self.mfm = mlcs.MultiFidelityModel(fidelities=['high', 'low'], archive=self.archive,
                                           kernel='Matern', scaling='off')

        self.utility = partial(UtilityFunction(kind='ei').utility, goal=self.goal)


    def iterate(self):  # sourcery skip: assign-if-exp
        reuse_fraction = 0  # only updated when (proto)EG is used
        iterations = 0
        start_time = time()
        while self.budget > 0:
            fidelity = self.select_fidelity()

            # select next best place to evaluate:
            if fidelity == 'high':
                x = self.select_next_high_fid()
            else:  # elif fidelity == 'low':
                x = self.select_next_low_fid()

            # evaluate best place
            y = self.func[fidelity](x.reshape(1, -1))[0]
            self.archive.addcandidate(candidate=x.flatten(), fitness=y, fidelity=fidelity)

            # update model & error grid
            self.mfm.retrain()
            if self.proto_eg and self.budget > 0:  # prevent unnecessary computation
                self.proto_eg.update_errorgrid_with_sample(x, fidelity=fidelity)
                reuse_fraction = self.proto_eg.reuse_fraction

            iterations += 1
            # logging
            self.log_entry(Entry(
                iteration=iterations,
                budget=self.budget,
                time_since_high_eval=self.time_since_high_eval,
                tau=-1,
                fidelity=fidelity,
                wall_time=time() - start_time,
                nhigh=self.archive.count('high'),
                nlow=self.archive.count('low'),
                reuse_fraction=reuse_fraction,
                candidate=x,
                fitness=y,
            ))
            np.save(self.run_save_dir / archive_file_template.format(iterations), self.archive)

        return self.mfm, pd.DataFrame.from_records(self.entries, columns=Entry._fields), self.archive


    def select_next_high_fid(self):
        # best predicted low-fid only datapoint for high-fid (to maintain hierarchical model)
        candidates = select_high_fid_only_candidates(self.archive)
        candidate_predictions = [
            (cand, self.utility(
                cand.reshape(1, -1),
                gp=self.mfm.top_level_model,
                y_best=self.archive.min['high']
                ), self.mfm.models['high'].predict(cand)
            )
            for cand in candidates
        ]

        x = max(candidate_predictions, key=itemgetter(1))[0].ravel()
        self.time_since_high_eval = 0
        self.budget -= 1
        return x


    def select_next_low_fid(self):
        self.time_since_high_eval += 1
        self.budget -= self.cost_ratio
        return self.acq_max(y_best=self.archive.max['high'], random_state=np.random.RandomState())


    def select_fidelity(self):

        if self.archive.count('high') >= self.archive.count('low'):
            return 'low'

        if self.fid_selection_method == 'EG':
            tau = calc_tau_from_EG(self.proto_eg.error_grid['mses'], self.cost_ratio)
            # compare \tau with current count t to select fidelity, must be >= 1
            fidelity = 'high' if 1 <= tau <= self.time_since_high_eval else 'low'
        elif self.fid_selection_method == 'fixed':
            fidelity = 'high' if 1 <= (1 / self.cost_ratio) <= self.time_since_high_eval else 'low'
        else:
            raise NotImplementedError(f"Fidelity selection method '{self.fid_selection_method}' has no implementation")

        return fidelity


    def log_entry(self, entry):

        self.entries.append(entry)
        with open(self.logfile, 'a') as csvfile:
            logwriter = writer(csvfile, delimiter=';')
            logwriter.writerow(entry)


    def acq_max(self, y_best, random_state, n_warmup=1_000, n_iter=50):
        """
        Code adapted from:
          https://github.com/fmfn/BayesianOptimization/blob/380b0d52ae0e3650b023c4ef6db43f7343c75dea/bayes_opt/util.py
        Under MIT License

        A function to find the maximum of the acquisition function
        It uses a combination of random sampling (cheap) and the 'L-BFGS-B'
        optimization method. First by sampling `n_warmup` (1e4) points at random,
        and then running L-BFGS-B from `n_iter` (50) random starting points.
        Parameters
        ----------
        :param y_best:
            The current maximum known value of the target function.
        :param random_state:
            instance of np.RandomState random number generator
        :param n_warmup:
            number of times to randomly sample the aquisition function
        :param n_iter:
            number of times to run scipy.minimize
        Returns
        -------
        :return: x_max, The arg max of the acquisition function.
        """

        bounds = self.func.bounds.T

        # Warm up with random points
        x_tries = random_state.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(n_warmup, bounds.shape[0])
        )
        ys = self.utility(x_tries, gp=self.mfm.top_level_model, y_best=y_best)
        # print(f'{x_tries.shape=}, {ys.shape=}')
        x_max = x_tries[ys.argmax()]
        max_acq = ys.max()

        # Explore the parameter space more throughly
        x_seeds = random_state.uniform(bounds[:, 0], bounds[:, 1],
                                       size=(n_iter, bounds.shape[0]))
        for x_try in x_seeds:
            # Find the minimum of minus the acquisition function
            res = minimize(
                lambda x: -self.utility(
                    x.reshape(1, -1),
                    gp=self.mfm.top_level_model,
                    y_best=y_best
                ).reshape(-1,),
                x_try.reshape(1, -1),
                bounds=bounds,
                method="L-BFGS-B",
            )

            # See if success
            if not res.success:
                continue

            # Store it if better than previous minimum(maximum).
            if max_acq is None or -res.fun[0] >= max_acq:
                x_max = res.x
                max_acq = -res.fun[0]

        # Clip output to make sure it lies within the bounds. Due to floating
        # point technicalities this is not always the case.
        return np.clip(x_max, bounds[:, 0], bounds[:, 1])


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


def class_fixed_ratio_multifid_bo(func, init_budget, cost_ratio, doe_n_high, doe_n_low, run_save_dir, seed_offset=None, **_):
    opt = Optimizer(func, init_budget, cost_ratio, doe_n_high, doe_n_low, run_save_dir, seed_offset=seed_offset, fid_selection_method='fixed')
    results = opt.iterate()
    return results


def class_proto_eg_multifid_bo(func, init_budget, cost_ratio, doe_n_high, doe_n_low, run_save_dir, nreps, seed_offset=None):
    opt = Optimizer(func, init_budget, cost_ratio, doe_n_high, doe_n_low, run_save_dir, num_reps=nreps, seed_offset=seed_offset, fid_selection_method='EG')
    results = opt.iterate()
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


def do_run(benchmark_func, run_save_dir, optimizer, kwargs):
    run_save_dir.mkdir(parents=True, exist_ok=True)
    optimizer(
        func=benchmark_func,
        run_save_dir=run_save_dir,
        **kwargs
    )


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
        mf2.invert(mf2.currin),  # 3
        mf2.himmelblau,          # 4
        mf2.six_hump_camelback,  # 5

        mf2.park91a,             # 6
        mf2.park91b,             # 7

        mf2.hartmann6,           # 8

        mf2.borehole,            # 9

        # mf2.forrester,           # 10
    ]

    if args.idx:
        functions = [functions[f_idx] for f_idx in args.idx]

    kwargs = {
        'init_budget': args.budget,
        'cost_ratio': args.cost_ratio,
        'doe_n_high': 5,
        'doe_n_low': 10,
        'num_reps': args.nreps,
    }

    optimizers = {
        'fixed': class_fixed_ratio_multifid_bo,
        'naive': simple_multifid_bo,
        'proto-eg': class_proto_eg_multifid_bo,
        # 'proto-eg': proto_EG_multifid_bo,
    }

    for func in functions:
        print(func.name)

        for idx, cost_ratio in product(range(args.niters), args.cost_ratio):
            kwargs['seed_offset'] = idx
            kwargs['cost_ratio'] = cost_ratio

            for name, optimizer in optimizers.items():
                if args.experiment not in [None, name]:
                    continue
                run_save_dir = save_dir / FOLDER_NAME_TEMPLATE.format(
                    func_name=func.name,
                    name=name,
                    cost_ratio=cost_ratio,
                    budget=args.budget,
                    idx=idx,
                )
                print(f'    {name} c{cost_ratio} b{args.budget} i{idx}...')
                do_run(func, run_save_dir, optimizer, kwargs)


if __name__ == '__main__':

    cost_ratios = [0.1, 0.2, 0.25, 0.5]

    parser = argparse.ArgumentParser()
    parser.add_argument('idx', type=int, nargs='*',
                        help='Experiment indices to run. Default: all')
    parser.add_argument('-e', '--experiment', type=str, nargs='?',
                        help='Experiment function to run. Options: fixed, naive, proto-eg. Default: all')
    parser.add_argument('--nreps', type=int, default=50,
                        help='number of independent repetitions to perform for the error grid')
    parser.add_argument('--niters', type=int, default=5,
                        help='number of independent iterations of the experiment to perform')
    parser.add_argument('-b', '--budget', type=int, default=100,
                        help='evaluation budget')
    parser.add_argument('-c', '--cost-ratio', type=float, default=cost_ratios, nargs='*',
                        help='relative cost of a low- vs high-fidelity evaluation')
    arguments = parser.parse_args()

    main(arguments)
