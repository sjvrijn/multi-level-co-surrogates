#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2021-09-09-mfbo-comparisons.py: script to plot comparisons between different
runs of (different) multi-fidelity bayesian optimization algorithms
"""

import argparse
from collections import defaultdict
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parse import compile
from pyprojroot import here

import processing as proc

data_path = here('files/2020-11-05-simple-mfbo/')
plot_path = here('plots/2021-09-09-mfbo-comparisons/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)

subfolder_template = compile('{func_name}-{method}-b{init_budget:d}-i{idx:d}')


def compare_different_runs(save_exts=('.png', '.pdf')):
    """Compare logged data from different runs for the same problem/strategy

    :param save_exts:  which extensions to use when saving plots
    """
    # read & group all subfolders that only differ by 'method' and 'idx'
    groups = defaultdict(list)
    for subfolder in data_path.iterdir():
        match = subfolder_template.parse(subfolder.name)
        if not match:
            continue
        group_id = (match['func_name'], match['init_budget'])
        groups[group_id].append((match['method'], match['idx'], subfolder))

    # for each group, create and plot a figure
    for (func_name, init_budget), folders in groups.items():
        print(f'{func_name} with init_budget={init_budget}')
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6), constrained_layout=True)
        fig.suptitle(
            f"Method comparison for {func_name} with init_budget={init_budget}"
        )
        # tracking colors per method for legend
        colors = {}
        # for each experiment, plot the data
        for method, idx, folder in sorted(folders, key=itemgetter(0)):
            df = pd.read_csv(folder / 'log.csv', index_col=0, sep=';')
            df = add_min_over_time_to_log(df, func_name.lower())
            if method in colors:
                label = ''
            else:
                label = method
                colors[method] = len(colors)
            plot_on_axes(axes, init_budget, df, label=label, color=f'C{colors[method]}', linewidth=.75)

        for suffix in save_exts:
            fig.savefig(plot_path / f'comparison-{func_name}-b{init_budget}{suffix}')
        plt.close()


def compare_different_strategies(save_exts=('.png', '.pdf')):
    """Compare logged data from different strategies for the same problem

    Assumes only a single run is available for now

    :param save_exts:  which extensions to use when saving plots
    """
    # read & group all subfolders that only differ by 'method'
    groups = defaultdict(list)
    for subfolder in data_path.iterdir():
        match = subfolder_template.parse(subfolder.name)
        if not match:
            continue
        group_id = (match['func_name'], match['init_budget'], match['idx'])
        groups[group_id].append((match['method'], subfolder))

    # for each group, create and plot a figure
    for (func_name, init_budget, idx), folders in groups.items():
        print(f'{func_name} with init_budget={init_budget} (idx {idx})')
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6), constrained_layout=True)
        fig.suptitle(
            f"Method comparison for {func_name} with init_budget={init_budget} (idx {idx})"
        )
        # for each experiment, plot the data
        for method, folder in sorted(folders, key=itemgetter(0)):
            df = pd.read_csv(folder / 'log.csv', index_col=0, sep=';')
            df = add_min_over_time_to_log(df, func_name.lower())
            plot_on_axes(axes, init_budget, df, label=method)

        for suffix in save_exts:
            fig.savefig(plot_path / f'comparison-{func_name}-b{init_budget}-i{idx}{suffix}')
        plt.close()


def plot_on_axes(axes, init_budget, df, label='', **kwargs):
    budget_used = init_budget - df['budget'].values

    ax = axes[0,0]
    # EG size path
    ax.plot(df['nlow'].values, df['nhigh'].values, marker='o', label=label, **kwargs)
    ax.set_title('EG size \'path\'')
    ax.set_ylabel('high-fid samples')
    ax.set_xlabel('low-fid samples')
    ax.legend(loc=0)

    ax = axes[0, 1]
    # tau / budget
    ax.plot(budget_used, df['tau'].values, label=label, **kwargs)
    ax.set_title('Tau')
    ax.set_ylim(bottom=-0.1, top=max(df['tau'].values + .1))
    ax.set_ylabel('$\\tau$')
    ax.set_xlabel('evaluation cost')
    ax.legend(loc='best')

    ax = axes[1, 0]
    # wall-time / budget
    ax.plot(budget_used, df['wall_time'].values, label=label, **kwargs)
    ax.set_title('wall-time')
    ax.set_yscale('log')
    ax.set_ylabel('time (s)')
    ax.set_xlabel('evaluation cost')
    ax.legend(loc='best')

    ax = axes[1, 1]
    # error to high-fidelity optimum for high-fid evaluated values
    ax.plot(budget_used, df['err_to_opt'], label=label, **kwargs)
    ax.set_title('distance to optimum')
    ax.set_yscale('log')
    ax.set_ylabel('y-error')
    ax.set_xlabel('evaluation cost')
    ax.legend(loc='best')


def add_min_over_time_to_log(df: pd.DataFrame, func_name: str):
    """Add the minimum fitness values over time for each fidelity to the dataframe"""

    # gather improvements per fidelity from the dataframe
    for fidelity in ['low', 'high']:
        fitnesses = np.array([np.inf] * len(df['fitness']))
        fitnesses[df['fidelity'] == fidelity] = df.loc[df['fidelity'] == fidelity]['fitness']
        fitnesses = np.minimum.accumulate(fitnesses)
        df[f'opt_{fidelity}'] = fitnesses

    func = proc.named_functions[func_name]
    df['err_to_opt'] = df['opt_high'] - func.high(func.x_opt)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exts", action="extend", nargs="+", type=str,
                        help="File extensions to use when saving images. Default: [.PNG, .PDF].")
    args = parser.parse_args()
    kwargs = {}
    if args.exts:
        kwargs['save_exts'] = args.exts
    # compare_different_strategies(**kwargs)
    compare_different_runs(**kwargs)
