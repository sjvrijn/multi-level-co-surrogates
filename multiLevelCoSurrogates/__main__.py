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

from .Surrogates import RBF, Kriging
from .Logger import Logger
from .config import data_dir, filename, suffix, data_ext, fit_funcs
from .config import experiment_repetitions, training_size


def createSurrogate(N, init_sample_size, fit_func, Surrogate):
    # The surrogate model starts by defining a sampling plan, we use an optimal Latin Hypercube here
    sp = samplingplan(N)
    init_candidates = sp.optimallhc(init_sample_size)
    results = np.array([fit_func(cand) for cand in init_candidates], ndmin=2).T

    # Now that we have our initial data, we can create an instance of a Kriging model
    surrogate = Surrogate(init_candidates, results)
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
        res = fit_func(candidates[index])
        results[index] = res
        archive_candidates.append((candidates[index], res))
    return results


def calcMultiFidelityError(candidate, highFidFunc, lowFidFunc):
    high = highFidFunc(candidate)
    low = lowFidFunc(candidate)
    return high - low



def runMultiFidelityExperiment(N, lambda_, lambda_pre, mu, init_sample_size,
                               fit_func_name, rep):

    ### SETUP ###
    Surrogate = Kriging
    fit_func = fit_funcs[fit_func_name]
    sigma = 0.5
    init_individual = N * [0]
    archive_candidates = []
    gen_counter = 0

    # Set up the filename detailing all settings of the experiment
    fname = filename.format(dim=N, func=fit_func_name)
    fsuff = suffix.format(size=training_size, rep=rep)
    filename_prefix = data_dir + 'benchmark_' + fname + fsuff

    pre_log = Logger(filename_prefix + 'prelog' + data_ext,
                     header="Pre-results, as predicted by the surrogate")
    res_log = Logger(filename_prefix + 'reslog' + data_ext,
                     header="Fitness values from actual function, inf for any not pre-selected candidate")
    full_res_log = Logger(filename_prefix + 'fullreslog' + data_ext,
                          header="Fitness values from actual function, evaluated for all candidates")

    error_func = partial(calcMultiFidelityError, highFidFunc=fit_func.high, lowFidFunc=fit_func.low)
    surrogate = createSurrogate(N, init_sample_size, error_func, Surrogate)
    es = cma.CMAEvolutionStrategy(init_individual, sigma, inopts={'popsize': lambda_pre, 'CMA_mu': mu, 'maxiter': 1000,
                                                                  'verb_filenameprefix': filename_prefix})

    if surrogate.name == 'Kriging':
        std_log = Logger(filename_prefix + 'stdlog' + data_ext,
                         header="Standard deviations associated with the pre-results,"
                                " as predicted by the Kriging surrogate")
    else:
        std_log = None


    ### OPTIMIZATION ###
    while not es.stop():
        # Obtain the list of lambda_pre candidates to evaluate
        candidates = np.array(es.ask())
        low_results = [fit_func.low(cand) for cand in candidates]
        low_errors = surrogate.predict(candidates)
        pre_results = [a + b for a, b in zip(low_results, low_errors)]

        results = preSelection(candidates, pre_results, lambda_, fit_func.high, archive_candidates)
        es.tell(candidates, results)
        full_res_log.writeLine([fit_func(cand) for cand in candidates])

        # Write data to disc to be plotted
        if std_log:
            pre_std = surrogate.predict_std(candidates)
            std_log.writeLine(pre_std)

        pre_log.writeLine(pre_results)
        res_log.writeLine(results)
        es.logger.add()
        # es.disp()

        gen_counter += 1
        #TODO: update retraining/archive to store both high and low fidelity values
        surrogate = retrain(archive_candidates, training_size, Surrogate)


def run():

    N = 10
    lambda_ = 4+int(3*np.log(N))
    lambda_pre = 2*lambda_
    mu = lambda_//2
    init_sample_size = 20

    num_reps = experiment_repetitions

    fit_func_names = fit_funcs.keys()

    experiments = product(range(num_reps), fit_func_names)
    for rep, fit_func_name in experiments:

        print("\n\n"
              "---------------------------------------------\n"
              "Function:           {fname}\n"
              "Repetittion:        {rep}".format(fname=fit_func_name, rep=rep))

        runMultiFidelityExperiment(N, lambda_, lambda_pre, mu, init_sample_size, fit_func_name, rep)


if __name__ == "__main__":
    run()
