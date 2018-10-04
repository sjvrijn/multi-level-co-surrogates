#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
main.py: This file contains the actual optimization experiments
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import os
import cma
import numpy as np
import time
from pathlib import Path
from pyKriging.samplingplan import samplingplan
from itertools import product

from multiLevelCoSurrogates.Surrogates import Surrogate, CoSurrogate
from multiLevelCoSurrogates.Logger import Logger
from multiLevelCoSurrogates.config import data_dir, folder_name, suffix, data_ext, fit_funcs, fit_func_dims
from multiLevelCoSurrogates.config import experiment_repetitions, training_sizes
from multiLevelCoSurrogates.BayesianOptimization import EGO
from multiLevelCoSurrogates.CandidateArchive import CandidateArchive


def guaranteeFolderExists(path_name):
    """ Make sure the given path exists after this call """
    path = Path(path_name)
    path.expanduser()
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
    except IndexError as e:
        if ndim < 2:
            sample = np.linspace(0, 1, num=init_sample_size)
        else:
            raise e

    sample = (sample*space) + l_bound
    return sample


def createSurrogate(N, init_sample_size, fit_func, l_bound, u_bound, surrogate_name):
    """
        Create a surrogate model on a Latin-Hypercube Sample within the given bounds.
        Returns the trained surrogate, and candidate archive containing these initial points

        :param N:                   Dimensionality, length of the desired vectors
        :param init_sample_size:    Number of samples to return
        :param fit_func:            Function to use for determining fitness of candidates
        :param l_bound:             Lower bound of the search space (numpy array)
        :param u_bound:             Upper bound of the search space (numpy array)
        :param surrogate_name:      Name of the surrogate type to use (e.g. Kriging)
        :return:                    Trained and initialized surrogate of desired type
    """

    init_candidates = createScaledLHS(N, init_sample_size, l_bound, u_bound)
    results = [fit_func(cand) for cand in init_candidates]
    results = np.array(results, ndmin=2).T

    cand_archive = CandidateArchive(ndim=N)
    for cand, res in zip(init_candidates, results):
        cand_archive.addcandidate(cand, res)

    # Now that we have our initial data, we can create an instance of the surrogate model
    surrogate = Surrogate.fromname(surrogate_name, cand_archive, n=init_sample_size)
    surrogate.train()
    return surrogate, cand_archive


def createCoSurrogate(N, init_sample_size, fit_func_low, fit_func_high, l_bound, u_bound, surrogate_name, fit_scaling_param=True):
    """
        Create a co-surrogate model on a Latin-Hypercube Sample within the given bounds.
        This model will be trained on the error between fidelities 'high' and 'low'.
        Returns the trained surrogate, and candidate archive containing these initial points

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

    cand_archive = CandidateArchive(ndim=N, fidelities=['high', 'low', 'high-low'])
    for cand, res_h, res_l in zip(init_candidates, results_high, results_low):
        cand_archive.addcandidate(cand, res_h, fidelity='high')
        cand_archive.addcandidate(cand, res_l, fidelity='low')

    # Now that we have our initial data, we can create an instance of the surrogate model
    surrogate = CoSurrogate(surrogate_name, cand_archive, fidelities=['high', 'low'], n=init_sample_size,
                            fit_scaling_param=fit_scaling_param)
    surrogate.train()
    return surrogate, cand_archive


def retrainMultiFidelity(cand_archive, training_size, surrogate_name, fit_scaling_param=True):
    """
    Retrieve the last `training_size` candidates from the archive and create a new co-surrogate trained on that dataset

    :param cand_archive:        CandidateArchive object
    :param training_size:       Number of most recent candidates to use to retrain
    :param surrogate_name:      Name of the surrogate type to re-initialize
    :param fit_scaling_param:   Train on error after a linear regression fit between the two levels? (default: True)
    :return:                    The newly retrained co-surrogate
    """
    co_surrogate = CoSurrogate(surrogate_name, cand_archive, fidelities=['high', 'low'], n=training_size,
                               fit_scaling_param=fit_scaling_param)
    co_surrogate.train()
    return co_surrogate


def retrain(cand_archive, training_size, surrogate_name):
    """
    Retrieve the last `training_size` candidates from the archive and create a new surrogate trained on that dataset

    :param cand_archive:        CandidateArchive object
    :param training_size:       Number of most recent candidates to use to retrain
    :param surrogate_name:      Name of the surrogate type to re-initialize
    :return:                    The newly retrained surrogate
    """
    surrogate = Surrogate.fromname(surrogate_name, cand_archive, n=training_size)
    surrogate.train()
    return surrogate


def multiFidelityPreSelection(candidates, pre_results, lambda_, fit_func, cand_archive):
    """
    Select the best `lambda_` candidates based on the `pre_results` and re-evaluate those using the actual fitness
    function. These candidates are added to the archive

    :param candidates:      Candidates of the current generation (Numpy array)
    :param pre_results:     Estimated fitnesses according to some pre-evaluation method
    :param lambda_:         Number of offspring to choose and evaluate on high fidelity
    :param fit_func:        The fitness function to use
    :param cand_archive:    CandidateArchive object
    :return:                High fidelity fitness results (list)
    """
    # Pre-selection evolution control: Choose the best lambda from lambda_pre to be re-evaluated (otherwise: np.inf)
    indices = np.argsort(pre_results)
    results = [np.inf for _ in candidates]
    for index in indices[:lambda_]:
        res_high = fit_func.high(candidates[index])
        res_low = fit_func.low(candidates[index])
        results[index] = res_high

        cand_archive.addcandidate(candidates[index], res_high, 'high')
        cand_archive.addcandidate(candidates[index], res_low, 'low')
    return results


def singleFidelityPreSelection(candidates, pre_results, lambda_, fit_func, cand_archive, fidelity=None):
    """
    Select the best `lambda_` candidates based on the `pre_results` and re-evaluate those using the actual fitness
    function. These candidates are added to the archive

    :param candidates:      Candidates of the current generation (Numpy array)
    :param pre_results:     Estimated fitnesses according to some pre-evaluation method
    :param lambda_:         Number of offspring to choose and evaluate on high fidelity
    :param fit_func:        The fitness function to use
    :param cand_archive:    CandidateArchive object
    :param fidelity:        Name of fidelity level to use when storing results in the candidate archive (optional)
    :return:                High fidelity fitness results (list)
    """
    # Pre-selection evolution control: Choose the best lambda from lambda_pre to be re-evaluated (otherwise: np.inf)
    indices = np.argsort(pre_results)
    results = [np.inf for _ in candidates]
    for index in indices[:lambda_]:
        res = fit_func(candidates[index])
        results[index] = res
        cand_archive.addcandidate(candidates[index], [res], fidelity=fidelity)
    return results



def runNoSurrogateExperiment(ndim, lambda_, mu, fit_func_name, rep):
    """
    Perform an optimization run on an optimization function using a regular CMA-ES


    :param ndim:            Dimensionality of the fitness function
    :param lambda_:         Offspring size of the CMA-ES
    :param mu:              Population size of the CMA-ES
    :param size:            <just an addition to the filename to make it easier for comparisons...>
    :param fit_func_name:   Name of the fitness function to use
    :param rep:             Repetition number
    """

    fit_func = fit_funcs[fit_func_name]
    sigma = 0.5
    init_individual = [(u + l) / 2 for u, l in zip(fit_func.u_bound, fit_func.l_bound)]
    l_bound = np.array(fit_func.l_bound)
    u_bound = np.array(fit_func.u_bound)

    # Set up the filename detailing all settings of the experiment
    fname = folder_name.format(ndim=ndim, func=fit_func_name, use='reg', surr='NoSurrogate')
    fsuff = suffix.format(size=0, rep=rep, gen=0)
    filename_prefix = f'{data_dir}{fname}{fsuff}'
    guaranteeFolderExists(f'{data_dir}{fname}')

    if f'{fsuff}reslog.{data_ext}' in os.listdir(f'{data_dir}{fname}'):
        return

    es = cma.CMAEvolutionStrategy(init_individual, sigma, inopts={'popsize': lambda_, 'CMA_mu': mu, 'maxiter': 1000,
                                                                  'verb_log': 0})

    res_log = Logger(f'{filename_prefix}reslog.{data_ext}',
                     header="Fitness values from actual function, inf for any not pre-selected candidate")
    time_log = Logger(f'{filename_prefix}timelog.{data_ext}',
                      header="Time spent on optimization process")
    start_time = time.time()

    while not es.stop():
        # Obtain the list of lambda_pre candidates to evaluate
        candidates = np.array([_keepInBounds(cand, l_bound, u_bound) for cand in es.ask()])
        results = [fit_func.high(cand) for cand in candidates]
        es.tell(candidates, results)
        res_log.writeLine(results)

    time_log.writeLine([time.time() - start_time])


def runExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, training_size,
                  fit_func_name, surrogate_name, rep, gen_interval=1):
    """
    Perform an optimization run on a given optimization function using a surrogate assisted CMA-ES

    :param ndim:                Dimensionality of the fitness function
    :param lambda_:             Offspring size of the CMA-ES
    :param lambda_pre:          Number of offspring to use in the pre-selection step
    :param mu:                  Population size of the CMA-ES
    :param init_sample_size:    Number of candidates to generate as an initial sample
    :param training_size:       Number of most recent candidates to use to retrain
    :param fit_func_name:       Name of the fitness function to use
    :param surrogate_name:      Name of the surrogate type to use
    :param rep:                 Repetition number
    """

    fit_func = fit_funcs[fit_func_name]
    sigma = 0.5
    init_individual = [(u+l)/2 for u, l in zip(fit_func.u_bound, fit_func.l_bound)]
    l_bound = np.array(fit_func.l_bound)
    u_bound = np.array(fit_func.u_bound)
    num_generations = 1

    # Set up the filename detailing all settings of the experiment
    fname = folder_name.format(ndim=ndim, func=fit_func_name, use='reg', surr=surrogate_name)
    fsuff = suffix.format(size=lambda_pre, rep=rep, gen=gen_interval)
    filename_prefix = f'{data_dir}{fname}{fsuff}'
    guaranteeFolderExists(f'{data_dir}{fname}')

    if f'{fsuff}reslog.{data_ext}' in os.listdir(f'{data_dir}{fname}'):
        return

    surrogate, cand_archive = createSurrogate(ndim, init_sample_size, fit_func.high, l_bound, u_bound, surrogate_name)
    es = cma.CMAEvolutionStrategy(init_individual, sigma, inopts={'popsize': lambda_pre, 'CMA_mu': mu, 'maxiter': 1000,
                                                                  'verb_log': 0})

    res_log = Logger(f'{filename_prefix}reslog.{data_ext}',
                     header="Fitness values from actual function, inf for any not pre-selected candidate")
    time_log = Logger(f'{filename_prefix}timelog.{data_ext}',
                      header="Time spent on optimization process")
    start_time = time.time()

    while not es.stop():
        # Obtain the list of lambda_pre candidates to evaluate
        candidates = np.array([_keepInBounds(cand, l_bound, u_bound) for cand in es.ask()])
        pre_results = surrogate.predict(candidates)

        if num_generations % gen_interval == 0:
            results = singleFidelityPreSelection(candidates, pre_results, lambda_, fit_func.high, cand_archive)
            res_log.writeLine(results)
        else:
            results = pre_results

        es.tell(candidates, results)

        surrogate = retrain(cand_archive, training_size, surrogate_name)
        num_generations += 1

    time_log.writeLine([time.time() - start_time])


def runEGOExperiment(ndim, init_sample_size, training_size, fit_func_name, surrogate_name, rep):
    """
    Perform an optimization run on an optimization function using an EGO approach

    :param ndim:                Dimensionality of the fitness function
    :param init_sample_size:    Number of candidates to generate as an initial sample
    :param training_size:       Number of most recent candidates to use to retrain
    :param fit_func_name:       Name of the fitness function to use
    :param surrogate_name:      Name of the surrogate type to use
    :param rep:                 Repetition number
    """

    from multiLevelCoSurrogates.post_process import make2dvisualizations

    num_iters = 100

    fit_func = fit_funcs[fit_func_name]
    l_bound = np.array(fit_func.l_bound)
    u_bound = np.array(fit_func.u_bound)

    # Set up the filename detailing all settings of the experiment
    fname = folder_name.format(ndim=ndim, func=fit_func_name, use='EGO-reg', surr=surrogate_name)
    fsuff = suffix.format(size=0, rep=rep, gen=0)
    filename_prefix = f'{data_dir}{fname}{fsuff}'
    guaranteeFolderExists(f'{data_dir}{fname}')

    # if f'{fsuff}reslog.{data_ext}' in os.listdir(f'{data_dir}{fname}'):
    #     return

    surrogate, cand_archive = createSurrogate(ndim, init_sample_size, fit_func.high, l_bound, u_bound, surrogate_name)
    ego = EGO(surrogate, ndim, fit_func.u_bound, fit_func.l_bound)

    res_log = Logger(f'{filename_prefix}reslog.{data_ext}',
                     header="Fitness values from actual function, inf for any not pre-selected candidate")

    for i in range(num_iters):

        print(i)
        make2dvisualizations(lambda x, y: ego.surrogate.predict([[x], [y]])[0],
                             l_bound, u_bound, f'{fit_func_name}_{surrogate_name}_landscape_{i}')

        x, ei = ego.next_infill()
        x_fit = fit_func.high(x)
        cand_archive.addcandidate(x, x_fit)
        ego.surrogate = retrain(cand_archive, training_size, surrogate_name)

        res_log.writeLine([x_fit])


def runMultiFidelityExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, training_size,
                               fit_func_name, surrogate_name, rep, fit_scaling_param=True, gen_interval=1):
    """
    Perform an optimization run on a multi-fidelity optimization function using a co-surrogate assisted CMA-ES

    :param ndim:                Dimensionality of the fitness function
    :param lambda_:             Offspring size of the CMA-ES
    :param lambda_pre:          Number of offspring to use in the pre-selection step
    :param mu:                  Population size of the CMA-ES
    :param init_sample_size:    Number of candidates to generate as an initial sample
    :param training_size:       Number of most recent candidates to use to retrain
    :param fit_func_name:       Name of the fitness function to use
    :param surrogate_name:      Name of the surrogate type to use
    :param rep:                 Repetition number
    :param fit_scaling_param:   Train on error after a linear regression fit between the two levels? (default: True)
    """

    ### SETUP ###
    fit_func = fit_funcs[fit_func_name]
    sigma = 0.5
    init_individual = [(u+l)/2 for u, l in zip(fit_func.u_bound, fit_func.l_bound)]
    l_bound = np.array(fit_func.l_bound)
    u_bound = np.array(fit_func.u_bound)
    num_generations = 1

    # Set up the filename detailing all settings of the experiment
    fname = folder_name.format(ndim=ndim, func=fit_func_name, use=f"{'scaled-MF' if fit_scaling_param else 'MF'}", surr=surrogate_name)
    fsuff = suffix.format(size=lambda_pre, rep=rep, gen=gen_interval)
    filename_prefix = f'{data_dir}{fname}{fsuff}'
    guaranteeFolderExists(f'{data_dir}{fname}')

    if f'{fsuff}reslog.{data_ext}' in os.listdir(f'{data_dir}{fname}'):
        return

    surrogate, cand_archive = createCoSurrogate(ndim, init_sample_size, fit_func.low, fit_func.high, l_bound, u_bound, surrogate_name, fit_scaling_param)
    es = cma.CMAEvolutionStrategy(init_individual, sigma, inopts={'popsize': lambda_pre, 'CMA_mu': mu, 'maxiter': 1000,
                                                                  'verb_log': 0})

    res_log = Logger(f'{filename_prefix}reslog.{data_ext}',
                     header="Fitness values from actual function, inf for any not pre-selected candidate")
    time_log = Logger(f'{filename_prefix}timelog.{data_ext}',
                      header="Time spent on optimization process")
    start_time = time.time()

    ### OPTIMIZATION ###
    while not es.stop():
        # Obtain the list of lambda_pre candidates to evaluate
        candidates = es.ask()
        candidates = np.array([_keepInBounds(cand, l_bound, u_bound) for cand in candidates])
        low_results = [fit_func.low(cand) for cand in candidates]
        low_errors = surrogate.predict(candidates)
        pre_results = [a*surrogate.rho + b for a, b in zip(low_results, low_errors)]

        if num_generations % gen_interval == 0:
            results = multiFidelityPreSelection(candidates, pre_results, lambda_, fit_func, cand_archive)
            res_log.writeLine(results)
        else:
            results = pre_results
        es.tell(candidates, results)

        surrogate = retrainMultiFidelity(cand_archive, training_size, surrogate_name, fit_scaling_param)
        num_generations += 1

    time_log.writeLine([time.time() - start_time])


def runBiSurrogateMultiFidelityExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, training_size,
                                          fit_func_name, surrogate_name, rep, fit_scaling_param=True, gen_interval=1):
    """
    Perform an optimization run on a multi-fidelity optimization function using a
    two-level (co-)surrogate assisted CMA-ES

    :param ndim:                Dimensionality of the fitness function
    :param lambda_:             Offspring size of the CMA-ES
    :param lambda_pre:          Number of offspring to use in the pre-selection step
    :param mu:                  Population size of the CMA-ES
    :param init_sample_size:    Number of candidates to generate as an initial sample
    :param training_size:       Number of most recent candidates to use to retrain
    :param fit_func_name:       Name of the fitness function to use
    :param surrogate_name:      Name of the surrogate type to use
    :param rep:                 Repetition number
    :param fit_scaling_param:   Train on error after a linear regression fit between the two levels? (default: True)
    """

    ### SETUP ###
    fit_func = fit_funcs[fit_func_name]
    sigma = 0.5
    init_individual = [(u+l)/2 for u, l in zip(fit_func.u_bound, fit_func.l_bound)]
    l_bound = np.array(fit_func.l_bound)
    u_bound = np.array(fit_func.u_bound)
    num_generations = 1

    # Set up the filename detailing all settings of the experiment
    fname = folder_name.format(ndim=ndim, func=fit_func_name,
                               use=f"{'scaled-MF-bisurr' if fit_scaling_param else 'MF-bisurr'}", surr=surrogate_name)
    fsuff = suffix.format(size=lambda_pre, rep=rep, gen=gen_interval)
    filename_prefix = f'{data_dir}{fname}{fsuff}'
    guaranteeFolderExists(f'{data_dir}{fname}')

    if f'{fsuff}reslog.{data_ext}' in os.listdir(f'{data_dir}{fname}'):
        return

    res_log = Logger(f'{filename_prefix}reslog.{data_ext}',
                     header="Fitness values from actual function, inf for any not pre-selected candidate")

    surrogate, cand_archive = createCoSurrogate(ndim, init_sample_size, fit_func.low, fit_func.high, l_bound, u_bound, surrogate_name, fit_scaling_param)
    surrogate_low = Surrogate.fromname(surrogate_name, cand_archive, n=training_size, fidelity='low')
    surrogate_low.train()

    es = cma.CMAEvolutionStrategy(init_individual, sigma, inopts={'popsize': lambda_pre*2, 'CMA_mu': mu, 'maxiter': 1000,
                                                                  'verb_log': 0})

    ### OPTIMIZATION ###
    while not es.stop():
        # Obtain the list of lambda_pre candidates to evaluate
        candidates = es.ask()
        candidates = np.array([_keepInBounds(cand, l_bound, u_bound) for cand in candidates])

        lowest_results = surrogate_low.predict(candidates)
        low_results = singleFidelityPreSelection(candidates, lowest_results, lambda_pre, fit_func.low, cand_archive, fidelity='low')
        low_errors = surrogate.predict(candidates)
        pre_results = [a*surrogate.rho + b for a, b in zip(low_results, low_errors)]

        if num_generations % gen_interval == 0:
            results = multiFidelityPreSelection(candidates, pre_results, lambda_, fit_func, cand_archive)
            res_log.writeLine(results)
        else:
            results = pre_results

        surrogate = retrainMultiFidelity(cand_archive, training_size, surrogate_name, fit_scaling_param)
        new_errors = surrogate.predict(candidates)
        new_results = [a*surrogate.rho + b for a, b in zip(low_results, new_errors)]
        for i, res in enumerate(results):
            if not np.isinf(res):
                results[i] = new_results[i]

        es.tell(candidates, results)
        num_generations += 1


def run():
    init_sample_size = 20

    fit_func_names = fit_funcs.keys()
    surrogates = ['RandomForest']  # , 'SVM''Kriging', 'RBF', ]
    lambda_pres = [2]  #, 4, 8]  # , 30, 50]
    gen_intervals = [1, 2, 3, 5, 10]  # , 20]
    experiments = product(lambda_pres, gen_intervals, range(1), fit_func_names, surrogates)

    for lambda_pre_mult, gen_int, rep, fit_func_name, surrogate_name in experiments:

        ndim = fit_func_dims[fit_func_name]
        lambda_ = 4 + int(3 * np.log(ndim))
        lambda_pre = lambda_pre_mult * lambda_
        mu = lambda_ // 2
        training_size = 0

        print(f"""--------------------------------------------------------------------------------
{lambda_pre}-{gen_int}: {surrogate_name} for {fit_func_name} ({rep}/{experiment_repetitions})""")

        # if surrogate_name == 'Kriging' and lambda_pre_mult == 2 and gen_int == 1:
        #     runNoSurrogateExperiment(ndim, lambda_, mu, fit_func_name, rep)

        # if surrogate_name in ['Kriging', 'RandomForest'] and lambda_pre == 10 and gen_int == 1:
        #     runEGOExperiment(ndim, init_sample_size, training_size, fit_func_name, surrogate_name, rep)

        # runExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, training_size, fit_func_name, surrogate_name, rep, gen_interval=gen_int)
        runMultiFidelityExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, training_size, fit_func_name, surrogate_name, rep, fit_scaling_param=True, gen_interval=gen_int)
        # runMultiFidelityExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, training_size, fit_func_name, surrogate_name, rep, fit_scaling_param=False, gen_interval=gen_int)

        # runBiSurrogateMultiFidelityExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, training_size, fit_func_name, surrogate_name, rep, fit_scaling_param=True, gen_interval=gen_int)
        # runBiSurrogateMultiFidelityExperiment(ndim, lambda_, lambda_pre, mu, init_sample_size, training_size, fit_func_name, surrogate_name, rep, fit_scaling_param=False, gen_interval=gen_int)



if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    run()
