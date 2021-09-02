#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-11-16-simple-mfbo.py: Some initial plotting of data gathered in .csv/.pkl
files during the simple-mfbo runs
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from parse import compile
from pyprojroot import here


data_path = here('files/2020-11-05-simple-mfbo/')
plot_path = here('plots/2020-11-16-simple-mfbo/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)

subfolder_template = compile('{func_name}-{method}-b{budget:d}-i{idx:d}')


def plot_on_axes(axes, budget_used, df, label=''):
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
    ax.set_ylim(bottom=0)
    ax.set_ylabel('time (s)')
    ax.set_xlabel('evaluation cost')

    ax = axes[1, 1]
    # reuse_fraction / budget
    ax.plot(budget_used, df['reuse_fraction'].values, label=label)
    ax.set_title('reuse_fraction')
    ax.set_ylim(bottom=0, top=1)
    ax.set_ylabel('model reuse fraction')
    ax.set_xlabel('evaluation cost')


def plot_comparison(file_a: Path, file_b: Path):

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6), constrained_layout=True)

    for file in [file_a, file_b]:
        match = tracking_template.parse(file.name)
        df = pd.read_csv(file, index_col=0)
        budget_used = match['budget'] - df['budget'].values
        plot_on_axes(axes, budget_used, df, label=match['method'])

    fig.suptitle(f"comparison for {match['name']} with budget={match['budget']} (idx {match['idx']})")
    for ax in axes.flatten():
        ax.legend(loc=0)
    # fig.show()
    fig.savefig(plot_path / file_a.with_suffix('.png').name.replace('naive', 'comparison'))


def make_all_plots_for_experiment_directory(experiment_dir: Path):

    # plot_title = f'{func.ndim}D {func.name} with {budget:.1f} budget left'
    # proto_eg.plot_errorgrid(
    #     title=plot_title,
    #     as_log=True,
    #     save_as=plot_dir / f'protoeg-EG-opt-{func.name}-{budget/cost_ratio:.0f}',
    #     save_exts=('png',),
    #     xlim=(3, init_budget),
    #     ylim=(2, (init_budget // 2)),
    # )
    # try:
    #     plot_archive(
    #         archive,
    #         func,
    #         title=plot_title,
    #         save_as=plot_dir / f'protoeg-archive-opt-{func.name}-{budget/cost_ratio:.0f}',
    #         save_exts=('png',),
    #     )
    # except NotImplementedError:
    #     pass
    ...


def plot_log(in_folder: Path, out_folder: Path, extensions=('png',)):
    match = subfolder_template.parse(in_folder.name)
    if not match:
        return

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6), constrained_layout=True)
    fig.suptitle(
        f"{match['method']} for {match['func_name']} with budget={match['budget']} (idx {match['idx']})")

    df = pd.read_csv(in_folder / 'log.csv', index_col=0, sep=';')
    try:
        budget_used = match['budget'] - df['budget'].values
    except KeyError:  # ignore for now
        return

    plot_on_axes(axes, budget_used, df)
    for ext in extensions:
        fig.savefig(out_folder / f'graphs.{ext}')
    plt.close()


def plot_and_gifify_archives(in_folder: Path, out_folder: Path):
    ...


def plot_and_gifify_errorgrids(in_folder: Path, out_folder: Path):
    ...



def perform_processing_for(experiment_folder: Path):
    """Run all relevant processing/plotting actions"""
    out_folder = plot_path / experiment_folder.name
    out_folder.mkdir(parents=True, exist_ok=True)

    plot_log(experiment_folder, out_folder)
    # plot_and_gifify_archives(experiment_folder, out_folder)
    # plot_and_gifify_errorgrids(experiment_folder, out_folder)



def main():
    for experiment_folder in data_path.iterdir():
        perform_processing_for(experiment_folder)


if __name__ == '__main__':
    main()
