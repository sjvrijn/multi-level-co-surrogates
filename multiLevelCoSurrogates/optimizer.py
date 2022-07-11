from operator import itemgetter

import pandas as pd
from scipy.optimize import minimize
from time import perf_counter

from functools import partial

from _csv import writer
from collections import namedtuple
from enum import IntEnum

import numpy as np
from pathlib import Path
from tqdm import tqdm

import mf2
import multiLevelCoSurrogates as mlcs

RANDOM_SEED_BASE = 20160501
RAND_SAMPLES_PER_DIM = 10
archive_file = 'archive.npz'
errorgrid_file_template = 'errorgrid_{:03d}.nc'
Entry = namedtuple('Entry', 'iteration budget time_since_high_eval tau fidelity wall_time nhigh nlow reuse_fraction candidate fitness')


class FidelitySelection(IntEnum):
    FIXED = 0
    NAIVE_EG = 1
    PROTO_EG = 2


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
            self.proto_eg = mlcs.ProtoEG(self.archive, num_reps=num_reps, interval=1)
            self.proto_eg.subsample_errorgrid()

        self.mfm = mlcs.MultiFidelityModel(fidelities=['high', 'low'], archive=self.archive,
                                           kernel='Matern', scaling='off')

        self.utility = partial(mlcs.UtilityFunction(kind='ei').utility, goal=self.goal)


    def iterate(self):  # sourcery skip: assign-if-exp
        reuse_fraction = 0  # only updated when (proto)EG is used
        iterations = 0
        eval_cost = {'high': 1, 'low': self.cost_ratio}
        start_time = perf_counter()
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
                    wall_time=perf_counter() - start_time,
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
            self.time_since_high_eval = 0
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
            self.tau = mlcs.calculate_tau(self.proto_eg.error_grid['mses'], self.cost_ratio)
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
    high_x, low_x = mlcs.scale_to_function(func, [high_x, low_x])
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
