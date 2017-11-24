#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

'''
main.py: This file contains the actual optimization experiments
'''

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import cma
import numpy as np
from pathlib import Path
from pyKriging.samplingplan import samplingplan
from itertools import product

from multiLevelCoSurrogates.Surrogates import Surrogate, CoSurrogate
from multiLevelCoSurrogates.Logger import Logger
from multiLevelCoSurrogates.config import data_dir, folder_name, suffix, data_ext, fit_funcs, fit_func_dims
from multiLevelCoSurrogates.config import experiment_repetitions, training_sizes
from multiLevelCoSurrogates.BayesianOptimization import EGO


def guaranteeFolderExists(path_name):
    """ Make sure the given path exists after this call """
    path = Path(path_name)
    path.mkdir(parents=True, exist_ok=True)


def _keepInBounds(x, l_bound, u_bound):
    """
        This function transforms x to t w.r.t. the low and high
        boundaries lb and ub. It implements the function T^{r}_{[a,b]} as
        described in Rui Li's PhD thesis "Mixed-Integer Evolution Strategies
        for Parameter Optimization and Their Applications to Medical Image
        Analysis" as alorithm 6.

        :param x:       Column vector to be kept in bounds
        :param l_bound: Lower bound column vector
        :param u_bound: Upper bound column vector
        :returns:       An in-bounds kept version of the column vector ``x``
    """

    y = (x - l_bound) / (u_bound - l_bound)
    floor_y = np.floor(y)

    I = np.mod(floor_y, 2) == 0
    yprime = np.zeros(np.shape(y))
    yprime[I] = np.abs(y[I] - floor_y[I])
    yprime[~I] = 1.0 - np.abs(y[~I] - floor_y[~I])

    x = l_bound + (u_bound - l_bound) * yprime
    return x


def createScaledLHS(ndim, init_sample_size, l_bound, u_bound):
    """
        Return a sample of `init_sample_size` points in `ndim` dimensions, scaled to cover the
        search space defined by `l_bound` and `u_bound`.

        :param ndim:                Dimensionality, length of the desired vectors
        :param init_sample_size:    Number of samples to return
        :param l_bound:             Lower bound of the search space (numpy array)
        :param u_bound:             Upper bound of the search space (numpy array)
        :return:                    Sample (2D numpy array)
    """
    space = u_bound - l_bound
    try:
        sp = samplingplan(ndim)
        sample = sp.optimallhc(init_sample_size)
        sample = (sample*space) + l_bound
    except IndexError as e:
        if ndim < 2:
            raise ValueError(f'LHS can only be defined for >= 2 dimensions (N={ndim} given)')
        else:
            raise e

    return sample


def createSurrogate(N, init_sample_size, fit_func, l_bound, u_bound, surrogate_name):
    """

        :param N:                   Dimensionality, length of the desired vectors
        :param init_sample_size:    Number of samples to return
        :param fit_func:            Function to use for determining fitness of candidates
        :param l_bound:             Lower bound of the search space (numpy array)
        :param u_bound:             Upper bound of the search space (numpy array)
        :param surrogate_name:      Name of the surrogate type to use (e.g. Kriging)
        :return:                    Trained and initialized surrogate of desired type
    """

    init_candidates = createScaledLHS(N, init_sample_size, l_bound, u_bound)
    results = np.array([fit_func(cand) for cand in init_candidates], ndmin=2).T

    # Now that we have our initial data, we can create an instance of the surrogate model
    surrogate = Surrogate.fromname(surrogate_name, init_candidates, results)
    surrogate.train()
    return surrogate


def createCoSurrogate(N, init_sample_size, fit_func_low, fit_func_high, l_bound, u_bound, surrogate_name, fit_scaling_param=True):
    """

        :param N:                   Dimensionality, length of the desired vectors
        :param init_sample_size:    Number of samples to return
        :param fit_func_low:        Low fidelity function to use for determining fitness of candidates
        :param fit_func_high:       High fidelity function to use for determining fitness of candidates
        :param l_bound:             Lower bound of the search space (numpy array)
        :param u_bound:             Upper bound of the search space (numpy array)
        :param surrogate_name:      Name of the surrogate type to use (e.g. Kriging)
        :return:                    Trained and initialized surrogate of desired type
    """

    init_candidates = createScaledLHS(N, init_sample_size, l_bound, u_bound)
    results_low = np.array([fit_func_low(cand) for cand in init_candidates], ndmin=2).T
    results_high = np.array([fit_func_high(cand) for cand in init_candidates], ndmin=2).T

    # Now that we have our initial data, we can create an instance of the surrogate model
    surrogate = CoSurrogate(surrogate_name, init_candidates, results_low, results_high, fit_scaling_param=fit_scaling_param)
    surrogate.train()
    return surrogate


def retrainMultiFidelity(archive_candidates_low, archive_candidates_high, training_size, surrogate_name, fit_scaling_param=True):
    x, y_low = zip(*archive_candidates_low[-training_size:])
    x, y_high = zip(*archive_candidates_high[-training_size:])
    co_surrogate = CoSurrogate(surrogate_name, np.array(list(x)),
                               np.array(list(y_low), ndmin=2).T, np.array(list(y_high), ndmin=2).T,
                               fit_scaling_param=fit_scaling_param)
    co_surrogate.train()
    return co_surrogate


def retrain(archive_candidates, training_size, surrogate_name):
    x, y = zip(*archive_candidates[-training_size:])
    surrogate = Surrogate.fromname(surrogate_name, np.array(list(x)), np.array(list(y), ndmin=2).T)
    surrogate.train()
    return surrogate


def multiFidelityPreSelection(candidates, pre_results, lambda_, fit_func, archive_candidates_low, archive_candidates_high):
    # Pre-selection evolution control: Choose the best lambda from lambda_pre to be re-evaluated (otherwise: np.inf)
    indices = np.argsort(pre_results)
    results = [np.inf for _ in candidates]
    for index in indices[:lambda_]:
        res_high = fit_func.high(candidates[index])
        res_low = fit_func.low(candidates[index])
        results[index] = res_high
        archive_candidates_low.append((candidates[index], res_low))
        archive_candidates_high.append((candidates[index], res_high))
    return results


def singleFidelityPreSelection(candidates, pre_results, lambda_, fit_func, archive_candidates):
    # Pre-selection evolution control: Choose the best lambda from lambda_pre to be re-evaluated (otherwise: np.inf)
    indices = np.argsort(pre_results)
    results = [np.inf for _ in candidates]
    for index in indices[:lambda_]:
        res = fit_func(candidates[index])
        results[index] = res
        archive_candidates.append((candidates[index], res))
    return results


def calcMultiFidelityError(candidate, highFidFunc, lowFidFunc):
    high = highFidFunc(candidate)
    low = lowFidFunc(candidate)
    return high - low


def create_loggers(surrogate, filename_prefix):
    pre_log = Logger(f'{filename_prefix}prelog.{data_ext}',
                     header="Pre-results, as predicted by the surrogate")
    res_log = Logger(f'{filename_prefix}reslog.{data_ext}',
                     header="Fitness values from actual function, inf for any not pre-selected candidate")
    full_res_log = Logger(f'{filename_prefix}fullreslog.{data_ext}',
                          header="Fitness values from actual function, evaluated for all candidates")
    if surrogate.provides_std:
        std_log = Logger(f'{filename_prefix}stdlog.{data_ext}',
                         header="Standard deviations associated with the pre-results,"
                                " as predicted by the Kriging surrogate")
    else:
        std_log = None

    return full_res_log, pre_log, res_log, std_log



def runMultiFidelityExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, training_size,
                               fit_func_name, surrogate_name, rep, fit_scaling_param=True):

    ### SETUP ###
    fit_func = fit_funcs[fit_func_name]
    sigma = 0.5
    init_individual = [(u+l)/2 for u, l in zip(fit_func.u_bound, fit_func.l_bound)]
    archive_candidates_low = []
    archive_candidates_high = []
    gen_counter = 0
    l_bound = np.array(fit_func.l_bound)
    u_bound = np.array(fit_func.u_bound)

    # Set up the filename detailing all settings of the experiment
    fname = folder_name.format(ndim=ndim, func=fit_func_name, use=f"{'scaled-MF' if fit_scaling_param else 'MF'}", surr=surrogate_name)
    fsuff = suffix.format(size=training_size, rep=rep)
    filename_prefix = f'{data_dir}{fname}{fsuff}'
    guaranteeFolderExists(f'{data_dir}{fname}')

    surrogate = createCoSurrogate(ndim, init_sample_size, fit_func.low, fit_func.high, l_bound, u_bound, surrogate_name, fit_scaling_param)
    es = cma.CMAEvolutionStrategy(init_individual, sigma, inopts={'popsize': lambda_pre, 'CMA_mu': mu, 'maxiter': 1000,
                                                                  'verb_filenameprefix': filename_prefix})

    full_res_log, pre_log, res_log, std_log = create_loggers(surrogate, filename_prefix)

    ### OPTIMIZATION ###
    while not es.stop():
        # Obtain the list of lambda_pre candidates to evaluate
        candidates = es.ask()
        candidates = np.array([_keepInBounds(cand, l_bound, u_bound) for cand in candidates])
        low_results = [fit_func.low(cand) for cand in candidates]
        low_errors = surrogate.predict(candidates)
        pre_results = [a*surrogate.rho + b for a, b in zip(low_results, low_errors)]

        results = multiFidelityPreSelection(candidates, pre_results, lambda_, fit_func, archive_candidates_low, archive_candidates_high)
        es.tell(candidates, results)
        full_res_log.writeLine([fit_func.high(cand) for cand in candidates])

        # Write data to disc to be plotted
        if std_log:
            pre_std = surrogate.predict(candidates, mode='std')
            std_log.writeLine(pre_std)

        pre_log.writeLine(pre_results)
        res_log.writeLine(results)
        es.logger.add()
        # es.disp()

        gen_counter += 1
        surrogate = retrainMultiFidelity(archive_candidates_low, archive_candidates_high, training_size, surrogate_name, fit_scaling_param)


def runExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, training_size,
                  fit_func_name, surrogate_name, rep):

    fit_func = fit_funcs[fit_func_name]
    sigma = 0.5
    init_individual = [(u+l)/2 for u, l in zip(fit_func.u_bound, fit_func.l_bound)]
    archive_candidates = []
    gen_counter = 0
    l_bound = np.array(fit_func.l_bound)
    u_bound = np.array(fit_func.u_bound)

    # Set up the filename detailing all settings of the experiment
    fname = folder_name.format(ndim=ndim, func=fit_func_name, use='reg', surr=surrogate_name)
    fsuff = suffix.format(size=training_size, rep=rep)
    filename_prefix = f'{data_dir}{fname}{fsuff}'
    guaranteeFolderExists(f'{data_dir}{fname}')

    surrogate = createSurrogate(ndim, init_sample_size, fit_func.high, l_bound, u_bound, surrogate_name)
    es = cma.CMAEvolutionStrategy(init_individual, sigma, inopts={'popsize': lambda_pre, 'CMA_mu': mu, 'maxiter': 1000,
                                                                  'verb_filenameprefix': filename_prefix})

    full_res_log, pre_log, res_log, std_log = create_loggers(surrogate, filename_prefix)

    while not es.stop():
        # Obtain the list of lambda_pre candidates to evaluate
        candidates = np.array([_keepInBounds(cand, l_bound, u_bound) for cand in es.ask()])
        pre_results = surrogate.predict(candidates)
        results = singleFidelityPreSelection(candidates, pre_results, lambda_, fit_func.high, archive_candidates)
        es.tell(candidates, results)
        full_res_log.writeLine([fit_func.high(cand) for cand in candidates])

        # Write data to disc to be plotted
        if std_log:
            pre_std = surrogate.predict(candidates, mode='std')
            std_log.writeLine(pre_std)

        pre_log.writeLine(pre_results)
        res_log.writeLine(results)
        es.logger.add()
        # es.disp()

        gen_counter += 1
        surrogate = retrain(archive_candidates, training_size, surrogate_name)


def runNoSurrogateExperiment(ndim, lambda_, mu, fit_func_name, rep):

    fit_func = fit_funcs[fit_func_name]
    sigma = 0.5
    init_individual = [(u+l)/2 for u, l in zip(fit_func.u_bound, fit_func.l_bound)]
    l_bound = np.array(fit_func.l_bound)
    u_bound = np.array(fit_func.u_bound)

    # Set up the filename detailing all settings of the experiment
    fname = folder_name.format(ndim=ndim, func=fit_func_name, use='reg', surr='NoSurrogate')
    fsuff = suffix.format(size=0, rep=rep)
    filename_prefix = f'{data_dir}{fname}{fsuff}'
    guaranteeFolderExists(f'{data_dir}{fname}')

    es = cma.CMAEvolutionStrategy(init_individual, sigma, inopts={'popsize': lambda_, 'CMA_mu': mu, 'maxiter': 1000,
                                                                  'verb_filenameprefix': filename_prefix})

    surrogate = Surrogate(None, None)
    full_res_log, pre_log, res_log, std_log = create_loggers(surrogate, filename_prefix)

    while not es.stop():
        # Obtain the list of lambda_pre candidates to evaluate
        candidates = np.array([_keepInBounds(cand, l_bound, u_bound) for cand in es.ask()])
        results = [fit_func.high(cand) for cand in candidates]
        es.tell(candidates, results)
        full_res_log.writeLine(results)
        res_log.writeLine(results)

        es.logger.add()


def runEGOExperiment(ndim, init_sample_size, training_size, fit_func_name, surrogate_name, rep):

    num_iters = 250

    fit_func = fit_funcs[fit_func_name]
    archive_candidates = []
    l_bound = np.array(fit_func.l_bound)
    u_bound = np.array(fit_func.u_bound)

    # Set up the filename detailing all settings of the experiment
    fname = folder_name.format(ndim=ndim, func=fit_func_name, use='EGO-reg', surr=surrogate_name)
    fsuff = suffix.format(size=training_size, rep=rep)
    filename_prefix = f'{data_dir}{fname}{fsuff}'
    guaranteeFolderExists(f'{data_dir}{fname}')

    surrogate = createSurrogate(ndim, init_sample_size, fit_func.high, l_bound, u_bound, surrogate_name)
    ego = EGO(surrogate, ndim, fit_func.u_bound, fit_func.l_bound)

    full_res_log, pre_log, res_log, std_log = create_loggers(surrogate, filename_prefix)

    for _ in range(num_iters):

        x, ei = ego.next_infill()
        x_fit = fit_func.high(x)
        archive_candidates.append((x, x_fit))
        ego.surrogate = retrain(archive_candidates, training_size, surrogate_name)

        res_log.writeLine([x_fit])
        full_res_log.writeLine([x_fit])


def run():
    init_sample_size = 20

    num_reps = experiment_repetitions
    fit_func_names = fit_funcs.keys()
    surrogates = ['Kriging', 'RBF', 'RandomForest']
    experiments = product(training_sizes, fit_func_names, surrogates, range(num_reps))

    for training_size, fit_func_name, surrogate_name, rep in experiments:

        ndim = fit_func_dims[fit_func_name]
        lambda_ = 4 + int(3 * np.log(ndim))
        lambda_pre = 2 * lambda_
        mu = lambda_ // 2

        print(f"""\n
              ---------------------------------------------
              Function:           {fit_func_name}
              Surrogate:          {surrogate_name}
              Repetittion:        {rep}""")

        # runNoSurrogateExperiment(ndim, lambda_, mu, fit_func_name, rep)

        runEGOExperiment(ndim, init_sample_size, training_size, fit_func_name, surrogate_name, rep)

        runExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, training_size, fit_func_name, surrogate_name, rep)
        runMultiFidelityExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, training_size, fit_func_name, surrogate_name, rep, fit_scaling_param=True)
        runMultiFidelityExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, training_size, fit_func_name, surrogate_name, rep, fit_scaling_param=False)


if __name__ == "__main__":
    run()
