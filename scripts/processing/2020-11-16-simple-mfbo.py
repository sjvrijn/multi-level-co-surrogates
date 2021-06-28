#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-11-16-simple-mfbo.py: Some initial plotting of data gathered in .csv/.pkl
files during the simple-mfbo runs
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mf2
from pathlib import Path
from parse import compile
from pyprojroot import here

data_path = here('files/2020-11-05-simple-mfbo/')
plot_path = here('plots/2020-11-16-simple-mfbo/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)

archive_template = compile('{name}-archive-{method}-b{budget:d}-i{idx:d}.pkl')
tracking_template = compile('{name}-tracking-{method}-b{budget:d}-i{idx:d}.csv')


def plot_tracking_for_file(tracking_file: Path):
    if not tracking_file.exists():
        return
    match = tracking_template.parse(tracking_file.name)
    if not match:
        return
    print(match['budget'])
    df = pd.read_csv(tracking_file, index_col=0)

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6), constrained_layout=True)
    budget_used = match['budget'] - df['budget'].values

    fig.suptitle(f"{match['method']} for {match['name']} with budget={match['budget']} (idx {match['idx']})")

    ax = axes[0,0]
    # EG size path
    ax.plot(df['nlow'].values, df['nhigh'].values, marker='o')
    ax.set_title('EG size \'path\'')
    ax.set_ylabel('high-fid samples')
    ax.set_xlabel('low-fid samples')

    ax = axes[0,1]
    # tau / budget
    ax.plot(budget_used, df['tau'].values)
    ax.set_title('Tau')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('$\tau$')
    ax.set_xlabel('evaluation cost')

    ax = axes[1,0]
    # wall-time / budget
    ax.plot(budget_used, df['wall_time'].values)
    ax.set_title('wall-time')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('time (s)')
    ax.set_xlabel('evaluation cost')

    ax = axes[1,1]
    # reuse_fraction / budget
    ax.plot(budget_used, df['reuse_fraction'].values)
    ax.set_title('reuse_fraction')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('model reuse fraction')
    ax.set_xlabel('evaluation cost')

    fig.show()




def plot_old_file_formats():
    # ## Plotting progress of $\tau$ over time

    for func in mf2.bi_fidelity_functions:
        tracking_file = data_path.joinpath(f'{func.name}-tracking.csv')
        if tracking_file.exists():
            print(func.name)
            df = pd.read_csv(tracking_file, index_col=0)
            df['tau'].plot()
            plt.savefig(plot_path / f'{func.name}-tau.png', bbox_inches='tight')
            #plt.show()
            plt.clf()

    # ## Plotting best high/low-fidelity evaluation over time
    # ### New-style: using `df[fitness]`

    for func in mf2.bi_fidelity_functions:
        tracking_file = data_path.joinpath(f'{func.name}-tracking.csv')
        if tracking_file.exists():
            print(func.name)
            df = pd.read_csv(tracking_file, index_col=0)
            #print(df.head())
            #print(df.dtypes)
            for name, sub_df in df.groupby('fidelity'):
                time = sub_df['budget'].to_numpy().reshape(-1,)
                #fitnesses = sub_df['fitness'].to_numpy().reshape(-1,)
                fitnesses = [eval(x)[0] for x in sub_df['fitness']]
                min_fit = np.minimum.accumulate(fitnesses)

                plt.plot(time, fitnesses, label=f'{name}-fidelity over time')
                plt.plot(time, min_fit, label=f'best {name}-fidelity over time')
            plt.legend(loc=0)
            plt.xlim([35, 0])
            plt.savefig(plot_path / f'{func.name}-evals.png', bbox_inches='tight')
            #plt.show()
            plt.clf()


def main():
    for f in data_path.iterdir():
        plot_tracking_for_file(f)


if __name__ == '__main__':
    main()
