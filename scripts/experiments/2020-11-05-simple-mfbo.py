#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Experiment file for comparing simple versions of multi-fidelity optimizers"""

import argparse
from csv import writer
from enum import IntEnum
from functools import partial
from itertools import product
from pathlib import Path
from time import time
from warnings import warn, simplefilter, catch_warnings

import mf2
import numpy as np
import pandas as pd
from collections import namedtuple
from operator import itemgetter
from scipy.optimize import minimize
from scipy.stats import norm
from pyprojroot import here
from tqdm import tqdm, TqdmWarning

from experiments import scale_to_function
import multiLevelCoSurrogates as mlcs


RANDOM_SEED_BASE = 20160501
RAND_SAMPLES_PER_DIM = 10

save_dir = here('files/2020-11-05-simple-mfbo/', warn=False)
save_dir.mkdir(parents=True, exist_ok=True)

FOLDER_NAME_TEMPLATE = '{func_name}-{name}-c{cost_ratio:f}-b{budget:d}-i{idx:d}'
archive_file = 'archive.npz'
errorgrid_file_template = 'errorgrid_{:03d}.nc'


Entry = namedtuple('Entry', 'iteration budget time_since_high_eval tau fidelity wall_time nhigh nlow reuse_fraction candidate fitness')


class FidelitySelection(IntEnum):
    FIXED = 0
    NAIVE_EG = 1
    PROTO_EG = 2


FIDELITY_SELECTORS = {
    'fixed': FidelitySelection.FIXED,
    'naive': FidelitySelection.NAIVE_EG,
    'proto-eg': FidelitySelection.PROTO_EG,
}

FUNCTIONS = [
    mf2.bohachevsky,  # 0
    mf2.booth,  # 1
    mf2.branin,  # 2
    mf2.invert(mf2.currin),  # 3
    mf2.himmelblau,  # 4
    mf2.six_hump_camelback,  # 5

    mf2.park91a,  # 6
    mf2.park91b,  # 7

    mf2.hartmann6,  # 8

    mf2.borehole,  # 9

    *[mf2.adjustable.branin(a) for a in np.round(np.linspace(0, 1, 11),2)],      # 10-20
    *[mf2.adjustable.paciorek(a) for a in np.round(np.linspace(0.1, 1, 10),2)],  # 21-30
    *[mf2.adjustable.hartmann3(a) for a in np.round(np.linspace(0, 1, 11),2)],   # 31-41
    *[mf2.adjustable.trid(a) for a in np.round(np.linspace(0, 1, 11),2)],        # 42-52
]


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
            err = f"The utility function {kind} has not been implemented, " \
                  f"please choose one of ucb, ei, or poi."
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

    def __init__(
        self,
        func,
        init_budget: int,
        cost_ratio: float,
        doe_n_high: int,
        doe_n_low: int,
        run_save_dir: Path,
        fid_selection_method: FidelitySelection,
        seed_offset: int=0,
        num_reps: int=50,
        goal: str='minimize',
        use_x_opt: bool=False,
    ):

        if doe_n_high + cost_ratio * doe_n_low >= init_budget:
            raise ValueError('Budget should not be exhausted after DoE')
        if not isinstance(fid_selection_method, FidelitySelection):
            raise ValueError('Invalid fidelity selection method, not in FidelitySelection')

        np.random.seed(RANDOM_SEED_BASE + seed_offset)

        self.func = func
        self.init_budget = init_budget
        self.cost_ratio = cost_ratio
        self.run_save_dir = run_save_dir
        self.fid_selection_method = fid_selection_method
        self.num_reps = num_reps
        self.goal = goal
        self.use_x_opt = use_x_opt
        self.time_since_high_eval = 0
        self.entries = []
        self.archive = make_mf_doe(func, doe_n_high, doe_n_low)
        # subtract mf-DoE from budget
        self.budget = self.init_budget - (doe_n_high + doe_n_low * cost_ratio)

        self.logfile = run_save_dir / 'log.csv'
        with open(self.logfile, 'w') as csvfile:
            logwriter = writer(csvfile, delimiter=';')
            logwriter.writerow(Entry._fields)

        self.proto_eg = None
        self.tau = 0
        if fid_selection_method in [FidelitySelection.NAIVE_EG, FidelitySelection.PROTO_EG]:
            self.proto_eg = mlcs.ProtoEG(self.archive, num_reps=num_reps)
            self.proto_eg.subsample_errorgrid()

        self.mfm = mlcs.MultiFidelityModel(fidelities=['high', 'low'], archive=self.archive,
                                           kernel='Matern', scaling='off')

        self.utility = partial(UtilityFunction(kind='ei').utility, goal=self.goal)


    def iterate(self):  # sourcery skip: assign-if-exp
        reuse_fraction = 0  # only updated when (proto)EG is used
        iterations = 0
        eval_cost = {'high': 1, 'low': self.cost_ratio}
        start_time = time()
        self.archive.save(self.run_save_dir / archive_file)
        if self.proto_eg:
            self.proto_eg.error_grid.to_netcdf(self.run_save_dir / errorgrid_file_template.format(iterations))

        if self.use_x_opt:
            y_opt = self.func.high(self.func.x_opt)

        with tqdm(total=self.init_budget, leave=False) as pbar:
            pbar.update(self.init_budget - self.budget)

            while self.budget > 0:
                fidelity = self.select_fidelity()

                # select next best place to evaluate:
                if fidelity == 'high':
                    x = self.select_next_high_fid()
                else:  # elif fidelity == 'low':
                    x, fidelity = self.select_next_low_fid()

                # evaluate best place
                self.budget -= eval_cost[fidelity]
                pbar.update(eval_cost[fidelity])
                y = self.func[fidelity](x.reshape(1, -1))[0]
                self.archive.addcandidate(candidate=x.flatten(), fitness=y, fidelity=fidelity)

                # update model & error grid
                self.mfm.retrain()
                if self.proto_eg:
                    self.proto_eg.error_grid.to_netcdf(self.run_save_dir / errorgrid_file_template.format(iterations))
                    if self.budget > 0:  # prevent unnecessary computation
                        if self.fid_selection_method == FidelitySelection.PROTO_EG:
                            self.proto_eg.update_errorgrid_with_sample(x, fidelity=fidelity)
                        elif self.fid_selection_method == FidelitySelection.NAIVE_EG:
                            self.proto_eg.subsample_errorgrid()
                        reuse_fraction = self.proto_eg.reuse_fraction

                iterations += 1
                # logging
                self.log_entry(Entry(
                    iteration=iterations,
                    budget=self.budget,
                    time_since_high_eval=self.time_since_high_eval,
                    tau=self.tau,
                    fidelity=fidelity,
                    wall_time=time() - start_time,
                    nhigh=self.archive.count('high'),
                    nlow=self.archive.count('low'),
                    reuse_fraction=reuse_fraction,
                    candidate=x,
                    fitness=y,
                ))
            self.archive.save(self.run_save_dir / archive_file)

            if fidelity == 'high' and self.use_x_opt and self.check_optimum_reached(y_opt):
                # shortcut the while-loop
                self.budget = 0

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
        return x


    def select_next_low_fid(self):
        """Suggest next sample by maximizing acquisition function on model.

        If the acquisition function suggests a previous sample again, there are
        two fallback options:
         - if the sample has not yet been evaluated in high-fidelity, the
           same sample is returned, but with fidelity switched to 'high'
         - if it has also already been evaluated in high-fidelity,
           `_suggest_low_from_random` is called to pick a random sample.
        """
        fidelity = 'low'
        self.time_since_high_eval += 1
        x = self.acq_max(y_best=self.archive.max['high'],
                         random_state=np.random.RandomState())

        if x not in self.archive:
            return x, fidelity

        # pass `x` as list and unpack result, since getfitnesses expects a list
        fitness = self.archive.getfitnesses([x], fidelity='high')[0]

        # `x` is not yet evaluated in high-fidelty, switch fidelity
        if np.isnan(fitness):
            return x, 'high'

        # `x` has also already been evaluated in high-fidelity, select new random instead
        x = self._suggest_low_fid_from_random()
        return x, fidelity


    def _suggest_low_fid_from_random(self, n_per_dim=RAND_SAMPLES_PER_DIM):
        """Suggest a new candidate: best according to model from random sample

        This method is intended as a fall-back option in case the regular
        low-fidelity selection procedure selects a candidate that is already
        present in the archive.
        """
        ndim = self.func.ndim

        while True:
            new_x = np.random.rand(n_per_dim*ndim, ndim)
            mlcs.rescale(new_x, range_in=(0,1), range_out=self.func.bounds)
            y = self.mfm.top_level_model.predict(new_x)
            x = new_x[np.argmin(y)]
            if x not in self.archive:
                return x


    def select_fidelity(self):

        if self.budget <= 1:
            return 'high'
        if self.budget <= 2:
            return 'low'
        if self.archive.count('high') >= self.archive.count('low'):
            return 'low'

        if self.fid_selection_method in [FidelitySelection.NAIVE_EG, FidelitySelection.PROTO_EG]:
            self.tau = calc_tau_from_EG(self.proto_eg.error_grid['mses'], self.cost_ratio)
            # compare \tau with current count t to select fidelity, must be >= 1
            fidelity = 'high' if 1 <= self.tau <= self.time_since_high_eval else 'low'
        elif self.fid_selection_method == FidelitySelection.FIXED:
            fidelity = 'high' if 1 <= (1 / self.cost_ratio) <= self.time_since_high_eval else 'low'
        else:
            msg = f"Fidelity selection method '{self.fid_selection_method}' has no implementation"
            raise NotImplementedError(msg)

        return fidelity


    def log_entry(self, entry):

        self.entries.append(entry)
        with open(self.logfile, 'a') as csvfile:
            logwriter = writer(csvfile, delimiter=';')
            logwriter.writerow(entry)


    def acq_max(self, y_best, random_state, n_warmup=1_000, n_iter=50):
        """Code adapted from:
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


    def check_optimum_reached(self, y_opt, tolerance=1e-6):
        if self.goal == 'minimize':
            best_found = self.archive.min['high']
        else:  # self.goal == 'maximize'
            best_found = self.archive.max['high']

        return abs(best_found - y_opt) < tolerance


def make_mf_doe(func: mf2.MultiFidelityFunction, doe_n_high: int, doe_n_low: int):
    # make mf-DoE
    high_x, low_x = mlcs.bi_fidelity_doe(func.ndim, doe_n_high, doe_n_low)
    high_x, low_x = scale_to_function(func, [high_x, low_x])
    high_y, low_y = func.high(high_x), func.low(low_x)
    # create archive
    return mlcs.CandidateArchive.from_bi_fid_doe(high_x, low_x, high_y, low_y)


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
    reg = mlcs.utils.error_grids.fit_lin_reg(EG)
    beta_1, beta_2 = reg.coef_[:2]
    # determine \tau based on beta_1, beta_2 and cost_ratio
    tau = np.ceil(1 / (beta_1 / (beta_2 / cost_ratio)))
    if tau <= 1:
        warn('Low-fidelity not expected to add information', category=TauSmallerThanOneWarning)
    return tau


def do_run(benchmark_func, fidelity_selector, kwargs, force_rerun=False):
    run_save_dir = kwargs['run_save_dir']
    run_save_dir.mkdir(parents=True, exist_ok=True)
    if force_rerun:
        for file in run_save_dir.iterdir():
            file.unlink()
    elif list(run_save_dir.iterdir()):
        return  # directory already contains files: don't overwrite

    optimizer = Optimizer(
        func=benchmark_func,
        fid_selection_method=fidelity_selector,
        **kwargs,
    )
    return optimizer.iterate()


def main(args):
    import sklearn
    simplefilter("ignore", category=FutureWarning)
    simplefilter("ignore", category=sklearn.exceptions.ConvergenceWarning)
    simplefilter("ignore", category=TauSmallerThanOneWarning)
    simplefilter("ignore", category=mlcs.LowHighFidSamplesWarning)
    simplefilter("ignore", category=TqdmWarning)

    functions = [FUNCTIONS[f_idx] for f_idx in args.idx]

    kwargs = {
        'init_budget': args.budget,
        'cost_ratio': args.cost_ratio,
        'doe_n_high': 5,
        'doe_n_low': 10,
        'num_reps': args.nreps,
        'use_x_opt': args.shortcut,
    }

    for func in functions:
        print(func.name)

        for idx, cost_ratio, experiment in product(range(args.niters), args.cost_ratio, args.experiment):
            kwargs['seed_offset'] = idx
            kwargs['cost_ratio'] = cost_ratio
            kwargs['run_save_dir'] = save_dir / FOLDER_NAME_TEMPLATE.format(
                func_name=func.name,
                name=experiment,
                cost_ratio=cost_ratio,
                budget=args.budget,
                idx=idx,
            )
            print(f'    {experiment} c{cost_ratio} b{args.budget} i{idx}...')
            do_run(func, FIDELITY_SELECTORS[experiment], kwargs, args.force_rerun)


if __name__ == '__main__':

    cost_ratios = [0.1, 0.2, 0.25, 0.5]
    experiments = ['fixed', 'naive', 'proto-eg']

    parser = argparse.ArgumentParser()
    parser.add_argument('idx', type=int, default=range(len(FUNCTIONS)), nargs='*',
                        help=f'Experiment indices [0-{len(FUNCTIONS) - 1}] to run. Default: all')
    parser.add_argument('-e', '--experiment', type=str, default=experiments, nargs='*',
                        help=f'Experiment function to run. Options: {", ".join(experiments)}. Default: all')
    parser.add_argument('--nreps', type=int, default=50,
                        help='number of independent repetitions to perform for the error grid')
    parser.add_argument('--niters', type=int, default=5,
                        help='number of independent iterations of the experiment to perform')
    parser.add_argument('-b', '--budget', type=int, default=100,
                        help='evaluation budget')
    parser.add_argument('-c', '--cost-ratio', type=float, default=cost_ratios, nargs='*',
                        help='relative cost of a low- vs high-fidelity evaluation')
    parser.add_argument('-f', '--force-rerun', action='store_true',
                        help='Force rerunning this experiment. Deletes previous files')
    parser.add_argument('--shortcut', action='store_true',
                        help="Stop optimization when optimum reached based on function's `x_opt`")
    arguments = parser.parse_args()

    # ensure only valid experiment names are passed on
    valid_experiments = []
    for exp in arguments.experiment:
        if exp in FIDELITY_SELECTORS:
            valid_experiments.append(exp)
        else:
            warn(f"Skipping experiment '{exp}', name not recognised.")
    arguments.experiment = valid_experiments

    main(arguments)
