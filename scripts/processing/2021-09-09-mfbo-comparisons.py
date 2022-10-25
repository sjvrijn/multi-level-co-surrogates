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
    for subfolder in sorted(data_path.iterdir()):
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
    for subfolder in sorted(data_path.iterdir()):
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
    _, cur_y_max = ax.get_ylim()
    new_y_max = max(max(df['tau'].values + .1), cur_y_max)
    ax.set_ylim(bottom=-0.1, top=new_y_max)
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


def plot_grouped_but_individually(save_exts=('.png', '.pdf')):
    """Compare logged data from different runs for the same problem/strategy.
    Creates separate plots for all relevant data instead of a combined figure.

    :param save_exts:  which extensions to use when saving plots
    """
    # read & group all subfolders that only differ by 'method' and 'idx'
    groups = defaultdict(list)
    for subfolder in sorted(data_path.iterdir()):
        match = subfolder_template.parse(subfolder.name)
        if not match:
            continue
        group_id = (match['func_name'], match['init_budget'])
        groups[group_id].append((match['method'], match['idx'], subfolder))

    # for each group, create and plot a figure
    for (func_name, init_budget), folders in groups.items():
        print(f'{func_name} with init_budget={init_budget}')

        fig1, ax1 = plt.subplots(figsize=(4,3), constrained_layout=True)
        fig2, ax2 = plt.subplots(figsize=(4,3), constrained_layout=True)
        fig3, ax3 = plt.subplots(figsize=(4,3), constrained_layout=True)
        fig4, ax4 = plt.subplots(figsize=(4,3), constrained_layout=True)

        axes = [ax1, ax2, ax3, ax4]
        figs = [fig1,fig2,fig3,fig4]
        names = ['eg-path', 'tau', 'wall-time', 'opt-err']
        plot_folders = []
        for name in names:
            plot_folder = plot_path / name
            plot_folder.mkdir(exist_ok=True)
            plot_folders.append(plot_folder)

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
                colors[method] = f'C{len(colors)}'

            budget_used = init_budget - df['budget'].values

            fmt = {
                'color': colors[method],
                'label': label,
                'linewidth': .75,
            }

            ax = axes[0]
            # EG size path
            ax.plot(df['nlow'].values, df['nhigh'].values, marker='o', **fmt)
            # ax.set_title('EG size \'path\'')
            ax.set_title(f'{func_name} with init_budget={init_budget} (idx {idx})')
            ax.set_ylabel('high-fid samples')
            ax.set_xlabel('low-fid samples')
            ax.legend(loc=0)

            ax = axes[1]
            # tau / budget
            ax.plot(budget_used, df['tau'].values, **fmt)
            # ax.set_title('Tau')
            ax.set_title(f'{func_name} with init_budget={init_budget} (idx {idx})')
            _, cur_y_max = ax.get_ylim()
            new_y_max = max(max(df['tau'].values + .1), cur_y_max)
            ax.set_ylim(bottom=-0.1, top=new_y_max)
            ax.set_ylabel('$\\tau$')
            ax.set_xlabel('evaluation cost')
            ax.legend(loc='best')

            ax = axes[2]
            # wall-time / budget
            ax.plot(budget_used, df['wall_time'].values, **fmt)
            # ax.set_title('wall-time')
            ax.set_title(f'{func_name} with init_budget={init_budget} (idx {idx})')
            ax.set_yscale('log')
            ax.set_ylabel('time (s)')
            ax.set_xlabel('evaluation cost')
            ax.legend(loc='best')

            ax = axes[3]
            # error to high-fidelity optimum for high-fid evaluated values
            ax.plot(budget_used, df['err_to_opt'], **fmt)
            # ax.set_title('distance to optimum')
            ax.set_title(f'{func_name} with init_budget={init_budget} (idx {idx})')
            ax.set_yscale('log')
            ax.set_ylabel('y-error')
            ax.set_xlabel('evaluation cost')
            ax.legend(loc='best')

        for suffix in save_exts:
            for fig, plot_type, plot_folder in zip(figs, names, plot_folders):
                fig.savefig(plot_folder / f'{plot_type}-comparison-{func_name}-b{init_budget}-{suffix}')
        plt.close('all')


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
    parser.add_argument("--singles", action=argparse.BooleanOptionalAction, default=False,
                        help="Plot every run comparison individually. Default: --no-singles.")
    parser.add_argument("--grouped", action=argparse.BooleanOptionalAction, default=True,
                        help="Plot comparison of methods over multiple runs. Default: --grouped.")
    parser.add_argument("--split-group", action=argparse.BooleanOptionalAction, default=False,
                        help="Plot comparison of methods over multiple runs, but in separate plots."
                        "Default: --no-split-group.")
    args = parser.parse_args()
    kwargs = {}
    if args.exts:
        kwargs['save_exts'] = args.exts

    if args.singles:
        compare_different_strategies(**kwargs)
    if args.grouped:
        compare_different_runs(**kwargs)
    if args.split_group:
        plot_grouped_but_individually(**kwargs)
