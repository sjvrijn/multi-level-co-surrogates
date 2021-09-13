#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2021-09-09-mfbo-comparisons.py: script to plot comparisons between different
runs of (different) multi-fidelity bayesian optimization algorithms
"""

import argparse
import sys
from collections import defaultdict
from operator import itemgetter

import matplotlib.pyplot as plt
import mf2
import numpy as np
import pandas as pd
from parse import compile
from pyprojroot import here

module_path = str(here())
if module_path not in sys.path:
    sys.path.append(module_path)

from multiLevelCoSurrogates import CandidateArchive

data_path = here('files/2020-11-05-simple-mfbo/')
plot_path = here('plots/2021-09-09-mfbo-comparisons/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)

subfolder_template = compile('{func_name}-{method}-b{init_budget:d}-i{idx:d}')
archive_template = compile('archive_{iteration:d}.npy')
errorgrid_template = compile('errorgrid_{iteration:d}.nc')

named_functions = {
    func.name.lower(): func
    for func in mf2.bi_fidelity_functions
}


def compare_different_runs(save_exts=('.png', '.pdf')):
    """Compare logged data from different runs for the same problem/strategy

    :param save_exts:  which extensions to use when saving plots
    """
    ...


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
        fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(8, 9), constrained_layout=True)
        fig.suptitle(
            f"Method comparison for {func_name} with init_budget={init_budget} (idx {idx})"
        )
        # for each experiment, plot the data
        for method, folder in sorted(folders, key=itemgetter(0)):
            df = pd.read_csv(folder / 'log.csv', index_col=0, sep=';')
            archive = np.load(folder / 'archive_000.npy', allow_pickle=True).item()
            df = add_min_over_time_to_log(df, archive)
            plot_on_axes(axes, init_budget, df, label=method)
        axes[0,0].legend(loc=0)

        for suffix in save_exts:
            fig.savefig(plot_path / f'comparison-{func_name}-b{init_budget}-i{idx}{suffix}')
        plt.close()


def plot_on_axes(axes, init_budget, df, label=''):
    budget_used = init_budget - df['budget'].values

    ax = axes[0,0]
    # EG size path
    ax.plot(df['nlow'].values, df['nhigh'].values, marker='o', label=label)
    ax.set_title('EG size \'path\'')
    ax.set_ylabel('high-fid samples')
    ax.set_xlabel('low-fid samples')

    ax = axes[0, 1]
    # tau / budget
    ax.plot(budget_used, df['tau'].values, label=label)
    ax.set_title('Tau')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('$\\tau$')
    ax.set_xlabel('evaluation cost')

    ax = axes[1, 0]
    # wall-time / budget
    ax.plot(budget_used, df['wall_time'].values, label=label)
    ax.set_title('wall-time')
    ax.set_yscale('log')
    ax.set_ylabel('time (s)')
    ax.set_xlabel('evaluation cost')

    ax = axes[1, 1]
    # reuse_fraction / budget
    ax.plot(budget_used, df['reuse_fraction'].values, label=label)
    ax.set_title('reuse_fraction')
    ax.set_ylim(bottom=0, top=1)
    ax.set_ylabel('model reuse fraction')
    ax.set_xlabel('evaluation cost')

    ax = axes[2,0]
    # fitness vs budget (regardless of fidelity)
    ax.plot(budget_used, df['fitness'])
    ax.set_title('evaluated fitness')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('fitness (high- and low-fidelity)')
    ax.set_xlabel('evaluation cost')

    ax = axes[2,1]
    # minimum fitness over time per fidelity
    ax.plot(budget_used, df['opt_low'], label=f'{label} (low)')
    ax.plot(budget_used, df['opt_high'], label=f'{label} (high)')
    ax.set_title('best fitness')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('fitness (high- and low-fidelity)')
    ax.set_xlabel('evaluation cost')
    ax.legend(loc='best')


def add_min_over_time_to_log(df: pd.DataFrame, init_archive: CandidateArchive):
    """Add the minimum fitness values over time for each fidelity to the dataframe"""

    # gather improvements per fidelity from the dataframe
    for fidelity in ['low', 'high']:
        fitnesses = np.array([np.inf] * len(df['fitness']))
        fitnesses[df['fidelity'] == fidelity] = df.loc[df['fidelity'] == fidelity]['fitness']
        fitnesses[0] = min(init_archive.min[fidelity], fitnesses[0])
        fitnesses = np.minimum.accumulate(fitnesses)
        df[f'opt_{fidelity}'] = fitnesses

    return df


def main(**kwargs):
    compare_different_strategies(**kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exts", action="extend", nargs="+", type=str)
    args = parser.parse_args()
    kwargs = {}
    if args.exts:
        kwargs['save_exts'] = args.exts
    main(**kwargs)
