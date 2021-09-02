#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-11-16-simple-mfbo.py: Some initial plotting of data gathered in .csv/.pkl
files during the simple-mfbo runs
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mf2
import pandas as pd
from parse import compile
from pyprojroot import here

import processing as proc
module_path = str(here())
if module_path not in sys.path:
    sys.path.append(module_path)

from multiLevelCoSurrogates import CandidateArchive

data_path = here('files/2020-11-05-simple-mfbo/')
plot_path = here('plots/2020-11-16-simple-mfbo/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)

subfolder_template = compile('{func_name}-{method}-b{budget:d}-i{idx:d}')
archive_template = compile('archive_{iteration:d}.pkl')

named_functions = {
    func.name.lower(): func
    for func in mf2.bi_fidelity_functions
}


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
    ...


def plot_log(in_folder: Path, out_folder: Path, save_exts=('png', 'pdf')) -> None:
    """Plot the most important logged data from {in_folder}/log.csv

    :param in_folder:  folder containing the log.csv file to read and plot
    :param out_folder: where to store resulting plots
    :param save_exts:  which extensions to use when saving plots
    """
    match = subfolder_template.parse(in_folder.name)
    if not match:
        raise ValueError(f"Folder name {in_folder.name} does not match expected template")

    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(8, 6), constrained_layout=True)
    fig.suptitle(
        f"{match['method']} for {match['func_name']} with budget={match['budget']} (idx {match['idx']})"
    )

    df = pd.read_csv(in_folder / 'log.csv', index_col=0, sep=';')
    budget_used = match['budget'] - df['budget'].values

    plot_on_axes(axes, budget_used, df)
    for ext in save_exts:
        fig.savefig(out_folder / f'graphs.{ext}')
    plt.close()


def plot_and_gifify_archives(in_folder: Path, out_folder: Path, save_exts=('png', 'pdf')):
    """Create 2d plots of all archive.pkl files in `in_folder`, including animated GIF

    :param in_folder:  folder containing the log.csv file to read and plot
    :param out_folder: where to store resulting plots
    :param save_exts:  which extensions to use when saving the individual plots
    """
    match = subfolder_template.parse(in_folder.name)
    func_name = match['func_name']

    archive_files = [
        (f, archive_template.parse(f.name)['iteration'])
        for f in in_folder.iterdir()
        if archive_template.parse(f.name)
    ]
    for archive_file, iteration_idx in archive_files:
        archive = CandidateArchive(ndim=0, fidelities=['high', 'low', 'high-low'])
        with open(archive_file, 'rb') as f:
            archive.data = pickle.load(f)
        proc.plot_archive(
            archive,
            named_functions[func_name.lower()],
            f'plot of archive at idx {iteration_idx}',
            save_as=out_folder / archive_file.name,
            save_exts=save_exts
        )

    # do_gifification_here()


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
