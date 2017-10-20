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
from pyKriging.samplingplan import samplingplan
from itertools import product
from functools import partial

from multiLevelCoSurrogates.Surrogates import Surrogate
from multiLevelCoSurrogates.Logger import Logger
from multiLevelCoSurrogates.config import data_dir, filename, suffix, data_ext, fit_funcs, fit_func_dims
from multiLevelCoSurrogates.config import experiment_repetitions, training_size


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


def createSurrogate(N, init_sample_size, fit_func, l_bound, u_bound, surrogate_name):
    space = u_bound - l_bound

    # The surrogate model starts by defining a sampling plan, we use an optimal Latin Hypercube here
    sp = samplingplan(N)
    init_candidates = sp.optimallhc(init_sample_size)
    init_candidates = [np.array(cand)*space + l_bound for cand in init_candidates]
    results = np.array([fit_func(cand) for cand in init_candidates], ndmin=2).T

    # Now that we have our initial data, we can create an instance of a Kriging model
    surrogate = Surrogate.fromname(surrogate_name, init_candidates, results)
    surrogate.train()
    return surrogate


def retrain(archive_candidates, training_size, Surrogate):
    x, y = zip(*archive_candidates[-training_size:])
    surrogate = Surrogate(np.array(list(x)), np.array(list(y), ndmin=2).T)
    surrogate.train()
    return surrogate


def preSelection(candidates, pre_results, lambda_, fit_func, archive_candidates):
    # Pre-selection evolution control: Choose the best lambda from lambda_pre to be re-evaluated (otherwise: np.inf)
    indices = np.argsort(pre_results)
    results = [np.inf for _ in candidates]
    for index in indices[:lambda_]:
        res_high = fit_func.high(candidates[index])
        res_low = fit_func.low(candidates[index])
        results[index] = res_high
        archive_candidates.append((candidates[index], res_high-res_low))
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



def runMultiFidelityExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size,
                               fit_func_name, surrogate_name, rep):

    ### SETUP ###
    fit_func = fit_funcs[fit_func_name]
    sigma = 0.5
    init_individual = [(u+l)/2 for u, l in zip(fit_func.u_bound, fit_func.l_bound)]
    archive_candidates = []
    gen_counter = 0
    l_bound = np.array(fit_func.l_bound)
    u_bound = np.array(fit_func.u_bound)

    # Set up the filename detailing all settings of the experiment
    fname = filename.format(ndim=ndim, func=fit_func_name)
    fsuff = suffix.format(size=training_size, rep=rep)
    filename_prefix = f'{data_dir}{fname}{fsuff}'

    error_func = partial(calcMultiFidelityError, highFidFunc=fit_func.high, lowFidFunc=fit_func.low)
    surrogate = createSurrogate(ndim, init_sample_size, error_func, l_bound, u_bound, surrogate_name)
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
        pre_results = [a + b for a, b in zip(low_results, low_errors)]

        results = preSelection(candidates, pre_results, lambda_, fit_func, archive_candidates)
        es.tell(candidates, results)
        full_res_log.writeLine([fit_func.high(cand) for cand in candidates])

        # Write data to disc to be plotted
        if std_log:
            pre_std = surrogate.predict_std(candidates)
            std_log.writeLine(pre_std)

        pre_log.writeLine(pre_results)
        res_log.writeLine(results)
        es.logger.add()
        # es.disp()

        gen_counter += 1
        surrogate = retrain(archive_candidates, training_size, surrogate_name)


def runExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size,
                  fit_func_name, surrogate_name, rep):

    fit_func = fit_funcs[fit_func_name]
    sigma = 0.5
    init_individual = [(u+l)/2 for u, l in zip(fit_func.u_bound, fit_func.l_bound)]
    archive_candidates = []
    gen_counter = 0
    l_bound = np.array(fit_func.l_bound)
    u_bound = np.array(fit_func.u_bound)

    # Set up the filename detailing all settings of the experiment
    fname = filename.format(ndim=ndim, func=fit_func_name)
    fsuff = suffix.format(size=training_size, rep=rep)
    filename_prefix = f'{data_dir}{fname}{fsuff}'

    surrogate = createSurrogate(ndim, init_sample_size, fit_func.high, l_bound, u_bound, surrogate_name)
    es = cma.CMAEvolutionStrategy(init_individual, sigma, inopts={'popsize': lambda_pre, 'CMA_mu': mu, 'maxiter': 1000,
                                                                  'verb_filenameprefix': filename_prefix})

    full_res_log, pre_log, res_log, std_log = create_loggers(surrogate, filename_prefix)

    while not es.stop():
        # Obtain the list of lambda_pre candidates to evaluate
        candidates = es.ask()
        candidates = np.array([_keepInBounds(cand, l_bound, u_bound) for cand in candidates])
        pre_results = surrogate.predict(candidates)
        results = singleFidelityPreSelection(candidates, pre_results, lambda_, fit_func.high, archive_candidates)
        es.tell(candidates, results)
        full_res_log.writeLine([fit_func.high(cand) for cand in candidates])

        # Write data to disc to be plotted
        if std_log:
            pre_std = surrogate.predict_std(candidates)
            std_log.writeLine(pre_std)

        pre_log.writeLine(pre_results)
        res_log.writeLine(results)
        es.logger.add()
        # es.disp()

        gen_counter += 1
        surrogate = retrain(archive_candidates, training_size, surrogate_name)


def run():
    init_sample_size = 20

    num_reps = experiment_repetitions
    fit_func_names = fit_funcs.keys()
    surrogates = ['Kriging', 'RBF', 'RandomForest']
    experiments = product(fit_func_names, surrogates, range(num_reps))

    for fit_func_name, surrogate_name, rep in experiments:

        ndim = fit_func_dims[fit_func_name]
        lambda_ = 4 + int(3 * np.log(ndim))
        lambda_pre = 2 * lambda_
        mu = lambda_ // 2

        print(f"""\n\n
              ---------------------------------------------\n
              Function:           {fit_func_name}\n
              Repetittion:        {rep}""")

        runExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, fit_func_name, surrogate_name, rep)


if __name__ == "__main__":
    run()
