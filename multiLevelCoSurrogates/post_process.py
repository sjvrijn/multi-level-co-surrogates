#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

'''
post_process.py: This file is intended to perform some simple post-processing
                 based on the data files generated by the CMA-ES.
'''

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from multiLevelCoSurrogates.config import experiment_repetitions, fit_funcs, fit_func_dims, folder_name, suffix, training_sizes, data_ext, plot_ext, data_dir, plot_dir
from multiLevelCoSurrogates.__main__ import guaranteeFolderExists
from itertools import product
from matplotlib import pyplot as plt
from collections import Counter

from pprint import pprint
import numpy as np

surrogates = ['Kriging', 'RBF', 'RandomForest', 'NoSurrogate']
uses = ['reg', 'MF', 'scaled-MF']#, 'EGO-reg']



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


def plotSimpleComparisons(training_size):

    num_reps = experiment_repetitions
    fit_func_names = fit_funcs.keys()
    # surrogates = ['Kriging', 'RBF', 'RandomForest']
    # uses = ['reg', 'MF', 'scaled-MF']
    experiments = product(fit_func_names, surrogates, range(num_reps))

    for fit_func_name, surrogate_name, rep in experiments:
        print(fit_func_name, surrogate_name, rep)

        ndim = fit_func_dims[fit_func_name]
        fsuff = suffix.format(size=training_size, rep=rep)
        plt.figure()

        for use in uses:

            if surrogate_name == 'NoSurrogate' and use is not 'reg':
                continue
            elif use == 'EGO-reg' and surrogate_name is not 'Kriging':
                continue

            fname = folder_name.format(ndim=ndim, func=fit_func_name, use=use, surr=surrogate_name)
            filename_prefix = f'{data_dir}{fname}{fsuff}'

            data = np.array(loadFitnessHistory(filename_prefix + 'reslog.' + data_ext, column=(1, -1)))
            data = np.ma.masked_invalid(data).min(axis=1)
            data = np.minimum.accumulate(data)
            plt.plot(data, label=use)

        plot_folder = folder_name.format(ndim=ndim, func=fit_func_name, use='', surr=surrogate_name)
        plot_name_prefix = f'{plot_dir}{plot_folder}{fsuff}'
        guaranteeFolderExists(f'{plot_dir}{plot_folder}')
        plt.xlabel('Evaluations')
        plt.ylabel('Fitness value')
        plt.legend(loc=0)
        plt.savefig(plot_name_prefix + 'reslog.' + plot_ext)
        plt.close()


def plotMedianComparisons(training_size):

    num_reps = experiment_repetitions
    fit_func_names = fit_funcs.keys()
    # surrogates = ['Kriging', 'RBF', 'RandomForest']
    # uses = ['reg', 'MF', 'scaled-MF']
    experiments = product(fit_func_names, surrogates)

    for fit_func_name, surrogate_name in experiments:
        print(fit_func_name, surrogate_name)

        ndim = fit_func_dims[fit_func_name]
        plt.figure()

        for use in uses:

            if surrogate_name == 'NoSurrogate' and use is not 'reg':
                continue
            elif use == 'EGO-reg' and surrogate_name is not 'Kriging':
                continue

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
        plt.xlabel('Evaluations')
        plt.ylabel('Fitness value')
        plt.yscale('log')
        plt.legend(loc=0)
        plt.savefig(plot_name_prefix + 'reslog.' + plot_ext)
        plt.close()


def counterToFilledTupleList(counter):
    result = []
    for usage in ['reg', 'MF', 'scaled-MF']:
        if usage not in counter:
            result.append((usage, 0))
        else:
            result.append((usage, counter[usage]))
    return result


def calcWinsPerStrategy(training_size):

    num_reps = experiment_repetitions
    fit_func_names = fit_funcs.keys()
    # surrogates = ['Kriging', 'RBF', 'RandomForest']
    # uses = ['reg', 'MF', 'scaled-MF']
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

    plt.figure(figsize=(4,4))
    plt.suptitle(f'Wins per usage type, training size = {training_size}')
    plt.subplot(211)
    # plt.title('Combined wins per usage type')
    # ids, vals = zip(*counterToFilledTupleList(c))
    # plt.hist(range(len(ids)), weights=vals)
    # plt.xticks(range(len(ids)), ids)

    ax1 = plt.subplot(211)
    ax1.set_title('by surrogate')
    indices = []
    values = []
    labels = []
    for name, counter in surr_c.items():
        ids, vals = zip(*counterToFilledTupleList(counter))
        indices.append(range(len(ids)))
        values.append(vals)
        labels.append(name)
    ax1.hist(indices, bins=np.arange(4)-.5, weights=values, label=labels, stacked=True, align='mid', rwidth=.5)
    ax1.set_xticks(range(len(ids)))
    ax1.set_xticklabels(ids)

    ax2 = plt.subplot(212)
    ax2.set_title('by benchmark function')
    indices = []
    values = []
    labels = []
    for name, counter in func_c.items():
        ids, vals = zip(*counterToFilledTupleList(counter))
        indices.append(range(len(ids)))
        values.append(vals)
        labels.append(name)
    ax2.hist(indices, bins=np.arange(4)-.5, weights=values, label=labels, stacked=True, align='mid', rwidth=.5)
    ax2.set_xticks(range(len(ids)))
    ax2.set_xticklabels(ids)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.tight_layout()
    #
    # box = ax1.get_position()
    # ax1.set_position([box.x0, box.y0, box.width * 0.65, box.height])
    # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #
    # box = ax2.get_position()
    # ax2.set_position([box.x0, box.y0, box.width * 0.65, box.height])
    # ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))


    fsuff = suffix.format(size=training_size, rep='')
    plot_name_prefix = f'{plot_dir}{fsuff}'
    guaranteeFolderExists(f'{plot_dir}')
    plt.savefig(plot_name_prefix + 'reslog-histogram_no-legend.' + plot_ext)


def plotBoxPlots(training_size):

    num_reps = experiment_repetitions
    fit_func_names = fit_funcs.keys()
    # surrogates = ['Kriging', 'RBF', 'RandomForest']
    # uses = ['reg', 'MF', 'scaled-MF']
    experiments = product(fit_func_names, surrogates)

    for fit_func_name, surrogate_name in experiments:
        print(fit_func_name, surrogate_name)

        plt.figure()
        ndim = fit_func_dims[fit_func_name]
        bests = []
        lengths = []

        for use in uses:

            if surrogate_name == 'NoSurrogate' and use is not 'reg':
                continue
            if use == 'EGO-reg' and surrogate_name is not 'Kriging':
                continue
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

        if surrogate_name == 'Kriging':
            labels = uses
        elif surrogate_name == 'NoSurrogate':
            labels = uses[:1]
        else:
            labels = uses[:3]

        plt.subplot(211)
        plt.title("Fitness values")
        plt.boxplot(bests, labels=labels)
        plt.yscale('log')

        plt.subplot(212)
        plt.title("Time to convergence")
        plt.boxplot(lengths, labels=labels)

        plt.tight_layout()
        fsuff = suffix.format(size=training_size, rep='')
        plot_folder = f'{fit_func_name}-{surrogate_name}-'
        plot_name_prefix = f'{plot_dir}{plot_folder}{fsuff}'
        guaranteeFolderExists(f'{plot_dir}')
        plt.savefig(plot_name_prefix + 'reslog-boxplot.' + plot_ext)
        plt.close()




def run():
    for training_size in training_sizes:
        print(training_size)
        # plotSimpleComparisons(training_size)
        # plotMedianComparisons(training_size)
        # calcWinsPerStrategy(training_size)
        plotBoxPlots(training_size)


if __name__ == '__main__':
    run()
