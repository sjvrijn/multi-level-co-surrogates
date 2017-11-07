#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

'''
post_process.py: This file is intended to perform some simple post-processing
                 based on the data files generated by the CMA-ES.
'''

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from multiLevelCoSurrogates.config import experiment_repetitions, fit_funcs, fit_func_dims, folder_name, suffix, training_size, data_ext, plot_ext, data_dir, plot_dir
from multiLevelCoSurrogates.__main__ import guaranteeFolderExists
from itertools import product
from matplotlib import pyplot as plt
from collections import Counter

from pprint import pprint
import numpy as np


def isTwoInts(value):
    """Returns True if the given value is a tuple or list consisting of exactly 2 integers"""
    return isinstance(value, (tuple, list)) \
           and len(value) == 2 \
           and all(isinstance(val, int) for val in value)


#TODO: rewrite to use slice()
def interpretColumn(column):
    """Interprets the 'column' argument of loadFitnessHistory into a start, end value pair"""
    if column is None:  # No specific column is requested, return everything
        start = 0
        end = None
    elif isinstance(column, int):  # One specific column is requested
        start = column
        end = column + 1
    elif isTwoInts(column):  # Multiple columns are requested
        start = column[0]
        end = column[1] + 1 if column[1] != -1 else None  # if -1 is given, we want up to and incl. the last column
    else:  # 'column' does not match expected format
        raise Exception("Invalid format for 'column': {col}".format(col=column))

    return start, end


def loadFitnessHistory(fname, column=None):
    """
    Return the data stored in the given filename as float values.
    Optionally, only data from a single, or range of columns can be selected.

    :param fname: The name of the file to retrieve the data from
    :param column:   Single or double integer to indicate desired column (optional)
    :return:         (Selected) data from the given file in 2D list format
    """

    start, end = interpretColumn(column)

    with open(fname, 'r') as f:
        next(f)  # Skip the first line which header information
        return [list(map(float, line.split(' ')[start:end])) for line in f]


#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================


def plotSimpleComparisons():

    num_reps = experiment_repetitions
    fit_func_names = fit_funcs.keys()
    surrogates = ['Kriging', 'RBF', 'RandomForest']
    uses = ['reg', 'MF', 'scaled-MF']
    experiments = product(fit_func_names, surrogates, range(num_reps))

    for fit_func_name, surrogate_name, rep in experiments:
        print(fit_func_name, surrogate_name, rep)

        ndim = fit_func_dims[fit_func_name]
        fsuff = suffix.format(size=training_size, rep=rep)
        plt.figure()

        for use in uses:
            fname = folder_name.format(ndim=ndim, func=fit_func_name, use=use, surr=surrogate_name)
            filename_prefix = f'{data_dir}{fname}{fsuff}'

            data = np.array(loadFitnessHistory(filename_prefix + 'reslog.' + data_ext, column=(1, -1)))
            data = np.ma.masked_invalid(data).min(axis=1)
            data = np.minimum.accumulate(data)
            plt.plot(data, label=use)

        plot_folder = folder_name.format(ndim=ndim, func=fit_func_name, use='', surr=surrogate_name)
        plot_name_prefix = f'{plot_dir}{plot_folder}{fsuff}'
        guaranteeFolderExists(f'{plot_dir}{plot_folder}')
        plt.legend(loc=0)
        plt.savefig(plot_name_prefix + 'reslog.' + plot_ext)
        plt.close()


def plotMedianComparisons():

    num_reps = experiment_repetitions
    fit_func_names = fit_funcs.keys()
    surrogates = ['Kriging', 'RBF', 'RandomForest']
    uses = ['reg', 'MF', 'scaled-MF']
    experiments = product(fit_func_names, surrogates)

    for fit_func_name, surrogate_name in experiments:
        print(fit_func_name, surrogate_name)

        ndim = fit_func_dims[fit_func_name]
        plt.figure()

        for use in uses:

            fname = folder_name.format(ndim=ndim, func=fit_func_name, use=use, surr=surrogate_name)
            total_data = []

            for rep in range(num_reps):
                fsuff = suffix.format(size=training_size, rep=rep)
                filename_prefix = f'{data_dir}{fname}{fsuff}'

                data = np.array(loadFitnessHistory(filename_prefix + 'reslog.' + data_ext, column=(1, -1)))
                data = np.ma.masked_invalid(data).min(axis=1)
                data = np.minimum.accumulate(data)
                total_data.append(data)

            min_idx = np.argmin(np.asarray([dat[-1] for dat in total_data]))
            plt.plot(total_data[min_idx], label=use)

        fsuff = suffix.format(size=training_size, rep='')
        plot_folder = f'{fit_func_name}-{surrogate_name}-'
        plot_name_prefix = f'{plot_dir}{plot_folder}{fsuff}'
        guaranteeFolderExists(f'{plot_dir}')
        plt.yscale('log')
        plt.legend(loc=0)
        plt.savefig(plot_name_prefix + 'reslog.' + plot_ext)
        plt.close()


def calcWinsPerStrategy():

    num_reps = experiment_repetitions
    fit_func_names = fit_funcs.keys()
    surrogates = ['Kriging', 'RBF', 'RandomForest']
    uses = ['reg', 'MF', 'scaled-MF']
    experiments = product(fit_func_names, surrogates)

    c = Counter()
    surr_c = {name: Counter() for name in surrogates}
    func_c = {name: Counter() for name in fit_func_names}

    for fit_func_name, surrogate_name in experiments:
        print(fit_func_name, surrogate_name)

        ndim = fit_func_dims[fit_func_name]
        best_res = {}

        for use in uses:

            fname = folder_name.format(ndim=ndim, func=fit_func_name, use=use, surr=surrogate_name)
            total_data = []

            for rep in range(num_reps):
                fsuff = suffix.format(size=training_size, rep=rep)
                filename_prefix = f'{data_dir}{fname}{fsuff}'

                data = np.array(loadFitnessHistory(filename_prefix + 'reslog.' + data_ext, column=(1, -1)))
                data = np.ma.masked_invalid(data).min(axis=1)
                data = np.minimum.accumulate(data)
                total_data.append(data)

            bests = np.asarray([dat[-1] for dat in total_data])
            best_res[use] = np.argsort(bests)[len(bests) // 2]

        if best_res['reg'] < best_res['MF'] and best_res['reg'] < best_res['scaled-MF']:
            c['reg'] += 1
            surr_c[surrogate_name]['reg'] += 1
            func_c[fit_func_name]['reg'] += 1
        elif best_res['MF'] < best_res['reg'] and best_res['MF'] < best_res['scaled-MF']:
            c['MF'] += 1
            surr_c[surrogate_name]['MF'] += 1
            func_c[fit_func_name]['MF'] += 1
        else:
            c['scaled-MF'] += 1
            surr_c[surrogate_name]['scaled-MF'] += 1
            func_c[fit_func_name]['scaled-MF'] += 1

    pprint(func_c)
    print()
    pprint(surr_c)
    print()
    pprint(c)


def plotBoxPlots():

    num_reps = experiment_repetitions
    fit_func_names = fit_funcs.keys()
    surrogates = ['Kriging', 'RBF', 'RandomForest']
    uses = ['reg', 'MF', 'scaled-MF']
    experiments = product(fit_func_names, surrogates)

    for fit_func_name, surrogate_name in experiments:
        print(fit_func_name, surrogate_name)

        plt.figure()
        ndim = fit_func_dims[fit_func_name]
        bests = []
        lengths = []

        for use in uses:

            fname = folder_name.format(ndim=ndim, func=fit_func_name, use=use, surr=surrogate_name)
            total_data = []
            total_lengths = []

            for rep in range(num_reps):
                fsuff = suffix.format(size=training_size, rep=rep)
                filename_prefix = f'{data_dir}{fname}{fsuff}'

                data = np.array(loadFitnessHistory(filename_prefix + 'reslog.' + data_ext, column=(1, -1)))
                total_lengths.append(len(data))
                data = np.ma.masked_invalid(data).min(axis=1)
                data = np.minimum.accumulate(data)
                total_data.append(data)

            bests.append(np.array([dat[-1] for dat in total_data]))
            lengths.append(total_lengths)

        plt.subplot(211)
        plt.title("Fitness values")
        plt.boxplot(bests, labels=uses)
        plt.yscale('log')

        plt.subplot(212)
        plt.title("Time to convergence")
        plt.boxplot(lengths, labels=uses)

        plt.tight_layout()
        fsuff = suffix.format(size=training_size, rep='')
        plot_folder = f'{fit_func_name}-{surrogate_name}-'
        plot_name_prefix = f'{plot_dir}{plot_folder}{fsuff}'
        guaranteeFolderExists(f'{plot_dir}')
        plt.savefig(plot_name_prefix + 'reslog-boxplot.' + plot_ext)
        plt.close()




def run():
    print(training_size)
    # plotSimpleComparisons()
    # plotMedianComparisons()
    # calcWinsPerStrategy()
    plotBoxPlots()


if __name__ == '__main__':
    run()
