#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

'''
post_process.py: This file is intended to perform some simple post-processing
                 based on the data files generated by the CMA-ES.
'''

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from multiLevelCoSurrogates.config import experiment_repetitions, fit_funcs, fit_func_dims, folder_name, suffix, data_ext, plot_ext, data_dir, plot_dir, base_dir
from multiLevelCoSurrogates.__main__ import guaranteeFolderExists
from itertools import product
from matplotlib import pyplot as plt
from collections import Counter, namedtuple

from pprint import pprint
import numpy as np
import pandas as pd

surrogates = ['Kriging', 'RBF', 'RandomForest', 'NoSurrogate']  # , 'SVM'
# uses = ['reg', 'EGO-reg', 'MF', 'scaled-MF', 'MF-bisurr', 'scaled-MF-bisurr']
uses = ['reg', 'EGO-reg', 'scaled-MF']
gen_intervals = [0, 1, 2, 3, 5, 10, 20]
lambda_pres = [0, 2]  # , 4, 8]
figsize = (6, 4.5)

Index = namedtuple('Index', ['fitfunc', 'surrogate', 'usage', 'repetition', 'genint', 'lambda_pre'])
TimingData = namedtuple('TimingData', ['function', 'surrogate', 'usage', 'repetition', 'gen_int', 'lambda_pre', 'time'])

x_lims = {
    'bohachevsky': 100,
    'branin': 500,
    'booth': 250,
    'himmelblau': 500,
    'sixHumpCamelBack': 750,
    'park91a': 500,
    'park91b': 500,
    'borehole': 300,
}


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
        next(f)  # Skip the first line which only contains header information
        return [list(map(float, line.split(' ')[start:end])) for line in f]



#TODO: Rewrite all plotting functions to plot from a list of 'Index' namedtuples
#=======================================================================================================================
#=======================================================================================================================
#=======================================================================================================================


def getdata():

    fit_func_names = fit_funcs.keys()
    experiments = product(fit_func_names, surrogates, uses, range(experiment_repetitions), gen_intervals, lambda_pres)

    data = {}

    for fit_func_name, surrogate_name, use, rep, gen_int, lambda_pre_mul in experiments:
        idx = Index(fit_func_name, surrogate_name, use, rep, gen_int, lambda_pre_mul)
        ndim = fit_func_dims[fit_func_name]
        lambda_pre = (4 + int(3 * np.log(ndim))) * lambda_pre_mul

        if surrogate_name == 'NoSurrogate' and use is not 'reg':
            continue
        elif use == 'EGO-reg' and surrogate_name not in ['Kriging', 'RandomForest']:
            continue

        fname = folder_name.format(ndim=ndim, func=fit_func_name, use=use, surr=surrogate_name)
        fsuff = suffix.format(size=lambda_pre, rep=rep, gen=gen_int)
        filename_prefix = f'{base_dir}data/{fname}{fsuff}'

        #TODO: better determine optimal values for each function
        try:
            data[idx] = np.array(loadFitnessHistory(filename_prefix + 'reslog.' + data_ext, column=(1, -1)))
            if fit_func_name == 'borehole':
                data[idx] *= -1
            elif fit_func_name == 'park91b':
                data[idx] -= 0.666666666666  # Manually extracted minimum found value...
            elif fit_func_name == 'branin':
                data[idx] += 320.731611436  # Manually extracted minimum found value...
        except:
            pass

    print("done")
    return data


def timingdatatocsv():

    fit_func_names = fit_funcs.keys()
    experiments = product(fit_func_names, surrogates, uses, range(experiment_repetitions), gen_intervals, lambda_pres)
    data = []

    for fit_func_name, surrogate_name, use, rep, gen_int, lambda_pre_mul in experiments:
        ndim = fit_func_dims[fit_func_name]
        lambda_pre = (4 + int(3 * np.log(ndim))) * lambda_pre_mul

        if surrogate_name == 'NoSurrogate' and use is not 'reg':
            continue
        elif use == 'EGO-reg' and surrogate_name not in ['Kriging', 'RandomForest']:
            continue

        fname = folder_name.format(ndim=ndim, func=fit_func_name, use=use, surr=surrogate_name)
        fsuff = suffix.format(size=lambda_pre, rep=rep, gen=gen_int)
        filename_prefix = f'{data_dir}{fname}{fsuff}'

        try:
            # In this case, we only ever expect a single value per file
            time = np.array(loadFitnessHistory(filename_prefix + 'timelog.' + data_ext, column=(1, -1)))[0][0]
            tup = TimingData(fit_func_name, surrogate_name, use, rep, gen_int, lambda_pre_mul, time)
            data.append(tup)
        except:
            pass

    df = pd.DataFrame(data, columns=TimingData._fields)
    df.to_csv(f'{data_dir}timing_summary.csv')

    print("done")


def getplottingvalues(total_data, min_perc=25, max_perc=75):

    max_len = max([len(dat) for dat in total_data])
    new_data = [dat.tolist() + [dat[-1]]*(max_len - len(dat)) for dat in total_data]
    new_data = np.stack(new_data)

    # Workaround to prevent negative values
    true_min = np.min(new_data)
    if true_min <= 0:
        positive = new_data > 0
        try:
            min_pos = np.min(new_data[positive])
        except Exception as e:
            print(new_data)
            print(new_data[positive])
        new_data[~positive] = min_pos

    minimum = np.percentile(new_data, min_perc, axis=0)
    mean = np.mean(new_data, axis=0)
    median = np.percentile(new_data, 50, axis=0)
    maximum = np.percentile(new_data, max_perc, axis=0)

    return minimum, mean, median, maximum


def compare_by_genint(data):
    """Create and save plots comparing the median convergence of `experiment_repetitions`
     runs for various uses of each surrogate"""

    fit_func_names = fit_funcs.keys()
    np.set_printoptions(precision=3, linewidth=2000)

    for fit_func_name, surrogate_name, use in product(fit_func_names, surrogates, uses):
        if surrogate_name == 'NoSurrogate' and use is not 'reg':
            continue
        elif use == 'EGO-reg' and surrogate_name not in ['Kriging', 'RandomForest']:
            continue

        plt.figure(figsize=figsize)
        num_plotted = 0

        for gen_int, lambda_pre in product(gen_intervals, lambda_pres):
            total_data = []

            for rep in range(experiment_repetitions):
                try:
                    idx = Index(fit_func_name, surrogate_name, use, rep, gen_int, lambda_pre)
                    dat = data[idx]
                except:
                    continue
                try:
                    dat = np.ma.masked_invalid(dat).min(axis=1)
                    dat = np.minimum.accumulate(dat)
                    total_data.append(dat)
                except Exception as e:
                    print(idx)
                    print(dat)

            if not total_data:
                continue
            minimum, mean, median, maximum = getplottingvalues(total_data)
            plt.plot(median, label='g_int='+str(gen_int))
            plt.fill_between(np.arange(len(minimum)), minimum, maximum, interpolate=True, alpha=0.2)
            num_plotted += 1

        if num_plotted <= 1:
            plt.close()
            continue

        guaranteeFolderExists(f'{plot_dir}by_genint/')
        plt.title(f'{fit_func_name}')
        plt.xlabel('Evaluations')
        plt.xlim(0, x_lims[fit_func_name])
        plt.ylabel('Fitness value')
        plt.yscale('log')
        plt.legend(loc=0)

        plt.tight_layout()
        plt.savefig(plot_dir + 'by_genint/' + fit_func_name + '-' + surrogate_name + '-' + use + '.' + plot_ext)
        plt.close()

    print("all plotted")


def compare_by_use(data):
    """Create and save plots comparing the median convergence of `experiment_repetitions`
     runs for various uses of each surrogate"""

    fit_func_names = fit_funcs.keys()
    np.set_printoptions(precision=3, linewidth=2000)

    for fit_func_name, gen_int_ in product(fit_func_names, gen_intervals):
        plt.figure(figsize=figsize)
        num_plotted = 0

        for surrogate_name, use, lambda_pre, gen_int in product(surrogates, uses, lambda_pres, [gen_int_]):
            total_data = []

            if surrogate_name == 'NoSurrogate' and use is not 'reg':
                continue
            elif use == 'EGO-reg' and surrogate_name not in ['Kriging', 'RandomForest']:
                continue

            for rep in range(experiment_repetitions):
                try:
                    idx = Index(fit_func_name, surrogate_name, use, rep, gen_int, lambda_pre)
                    dat = data[idx]
                except:
                    continue
                try:
                    dat = np.ma.masked_invalid(dat).min(axis=1)
                    dat = np.minimum.accumulate(dat)
                    total_data.append(dat)
                except Exception as e:
                    print(idx)
                    print(dat)

            if not total_data:
                continue
            minimum, mean, median, maximum = getplottingvalues(total_data)
            plt.plot(median, label=f'{surrogate_name} - {"c" if use != "reg" else ""}SA-CMA-ES')
            plt.fill_between(np.arange(len(minimum)), minimum, maximum, interpolate=True, alpha=0.2)
            num_plotted += 1

        if num_plotted <= 1:
            plt.close()
            continue

        guaranteeFolderExists(f'{plot_dir}by_use/')
        plt.title(f'{fit_func_name}')
        plt.xlabel('Evaluations')
        plt.xlim(0, x_lims[fit_func_name])
        plt.ylabel('Fitness value')
        plt.yscale('log')
        plt.legend(loc=0)

        plt.tight_layout()
        plt.savefig(plot_dir + 'by_use/' + fit_func_name + '-' + '-' + str(gen_int_) + '.' + plot_ext)
        plt.close()

    print("all plotted")


def compare_by_surrogate(data):
    """Create and save plots comparing the median convergence of `experiment_repetitions`
     runs for various uses of each surrogate"""

    fit_func_names = fit_funcs.keys()
    np.set_printoptions(precision=3, linewidth=2000)

    for fit_func_name, use, gen_int_ in product(fit_func_names, uses, gen_intervals):
        plt.figure(figsize=figsize)
        num_plotted = 0

        for surrogate_name, lambda_pre, gen_int in product(surrogates, lambda_pres, [0, gen_int_]):
            total_data = []

            if use == 'EGO-reg' and surrogate_name not in ['Kriging', 'RandomForest']:
                continue

            for rep in range(experiment_repetitions):
                try:
                    idx = Index(fit_func_name, surrogate_name, use, rep, gen_int, lambda_pre)
                    dat = data[idx]
                except:
                    continue
                try:
                    dat = np.ma.masked_invalid(dat).min(axis=1)
                    dat = np.minimum.accumulate(dat)
                    total_data.append(dat)
                except Exception as e:
                    print(idx)
                    print(dat)

            if not total_data:
                continue

            minimum, mean, median, maximum = getplottingvalues(total_data)
            plt.plot(median, label=f'{surrogate_name}')
            plt.fill_between(np.arange(len(minimum)), minimum, maximum, interpolate=True, alpha=0.2)
            num_plotted += 1

        if num_plotted <= 1:
            plt.close()
            continue

        guaranteeFolderExists(f'{plot_dir}by_surrogate/')
        plt.title(f'{fit_func_name}')
        plt.xlabel('Evaluations')
        plt.xlim(0, x_lims[fit_func_name])
        plt.ylabel('Fitness value')
        plt.yscale('log')
        plt.legend(loc=0)

        plt.tight_layout()
        plt.savefig(plot_dir + 'by_surrogate/' + fit_func_name + '-' + use + '-' + str(gen_int) + '.' + plot_ext)
        plt.close()

    print("all plotted")


def make2dvisualizations(func, l_bound, u_bound, name, num_intervals=100):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    func = np.vectorize(func)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    x_min, y_min = l_bound
    x_max, y_max = u_bound

    # Make data.
    X = np.arange(x_min, x_max, (x_max-x_min)/num_intervals)
    Y = np.arange(y_min, y_max, (y_max-y_min)/num_intervals)
    X, Y = np.meshgrid(X, Y)
    Z = func(X, Y)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.view_init(azim=45)
    ax.set_title(f'{name}')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f')

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.tight_layout()
    guaranteeFolderExists(f'{plot_dir}surfaces/')
    plt.savefig(f'{plot_dir}surfaces/{name}_2d.{plot_ext}')


def run():
    data = getdata()
    compare_by_use(data)
    compare_by_genint(data)
    compare_by_surrogate(data)

    for fit_func_name in list(fit_funcs.keys())[:5]:

        l_bound = np.array(fit_funcs[fit_func_name].l_bound, dtype=np.float64)
        u_bound = np.array(fit_funcs[fit_func_name].u_bound, dtype=np.float64)

        func = lambda x, y: fit_funcs[fit_func_name].high((x,y))
        make2dvisualizations(func, l_bound, u_bound, fit_func_name)
        func = lambda x, y: fit_funcs[fit_func_name].low((x,y))
        make2dvisualizations(func, l_bound, u_bound, fit_func_name + '_low')
    #
    #     l_bound /= 33
    #     u_bound /= 33
    #
    #     func = lambda x, y: fit_funcs[fit_func_name].high((x,y))
    #     make2dvisualizations(func, l_bound, u_bound, fit_func_name + '_small')
    #     func = lambda x, y: fit_funcs[fit_func_name].low((x,y))
    #     make2dvisualizations(func, l_bound, u_bound, fit_func_name + '_small_low')


    timingdatatocsv()


if __name__ == '__main__':
    run()
