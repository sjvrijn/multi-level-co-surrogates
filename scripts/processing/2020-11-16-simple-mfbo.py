#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-11-16-simple-mfbo.py: Some initial plotting of data gathered in .csv/.pkl
files during the simple-mfbo runs
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mf2
import numpy as np
import pandas as pd
import xarray as xr
from parse import compile
from pyprojroot import here

import processing as proc

module_path = str(here())
if module_path not in sys.path:
    sys.path.append(module_path)

import multiLevelCoSurrogates as mlcs
from multiLevelCoSurrogates import CandidateArchive

data_path = here('files/2020-11-05-simple-mfbo/')
plot_path = here('plots/2020-11-16-simple-mfbo/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)

subfolder_template = compile('{func_name}-{method}-b{init_budget:d}-i{idx:d}')
archive_template = compile('archive_{iteration:d}.npy')
errorgrid_template = compile('errorgrid_{iteration:d}.nc')

named_functions = {
    func.name.lower(): func
    for func in mf2.bi_fidelity_functions
}

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


def calc_num_pixels(num_high, num_low):
    return np.array([
        len(mlcs.InstanceSpec(h, l, num_reps=1))
        for h, l in zip(num_high, num_low)
    ])


def plot_log(in_folder: Path, out_folder: Path, save_exts=('.png', '.pdf')) -> None:
    """Plot the most important logged data from {in_folder}/log.csv

    :param in_folder:  folder containing the log.csv file to read and plot
    :param out_folder: where to store resulting plots
    :param save_exts:  which extensions to use when saving plots
    """
    match = subfolder_template.parse(in_folder.name)
    if not match:
        raise ValueError(f"Folder name {in_folder.name} does not match expected template")

    fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(8, 9), constrained_layout=True)
    fig.suptitle(
        f"{match['method']} for {match['func_name']} with init_budget={match['init_budget']} (idx {match['idx']})"
    )
    df = pd.read_csv(in_folder / 'log.csv', index_col=0, sep=';')
    plot_on_axes(axes, match['init_budget'], df)
    for suffix in save_exts:
        fig.savefig(out_folder / f'graphs{suffix}')
    plt.close()


def plot_and_gifify_archives(in_folder: Path, out_folder: Path, save_exts=('.png', '.pdf')):
    """Create 2d plots of all archive.npy files in `in_folder`, including animated GIF

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
        archive = np.load(archive_file, allow_pickle=True).item()
        proc.plot_archive(
            archive,
            func=named_functions[func_name.lower()],
            title=f'plot of archive at idx {iteration_idx}',
            save_as=out_folder / archive_file.stem,
            suffixes=save_exts
        )
    proc.gifify_in_folder(out_folder, base_name='archive')


def plot_and_gifify_errorgrids(in_folder: Path, out_folder: Path, save_exts=('png', 'pdf')):
    """Create errorgrid plots errorgrid.nc files in `in_folder`, including animated GIF

    :param in_folder:  folder containing the log.csv file to read and plot
    :param out_folder: where to store resulting plots
    :param save_exts:  which extensions to use when saving the individual plots
    """
    match = subfolder_template.parse(in_folder.name)
    errorgrid_files = [
        (f, errorgrid_template.parse(f.name)['iteration'])
        for f in in_folder.iterdir()
        if errorgrid_template.parse(f.name)
    ]
    for errorgrid_file, iteration_idx in errorgrid_files:
        errorgrid = xr.load_dataarray(errorgrid_file)
        proc.plot_error_grid(
            errorgrid,
            title=f'plot of errorgrid at iteration {iteration_idx}',
            as_log=True,
            xlim=(3, match['init_budget']),
            ylim=(2, (match['init_budget'] // 2)),
            save_as=out_folder / errorgrid_file.stem,
        )
    proc.gifify_in_folder(out_folder, base_name='errorgrid')


def perform_processing_for(experiment_folder: Path, **kwargs):
    """Run all relevant processing/plotting actions"""
    out_folder = plot_path / experiment_folder.name
    out_folder.mkdir(parents=True, exist_ok=True)

    plot_log(experiment_folder, out_folder, **kwargs)
    plot_and_gifify_errorgrids(experiment_folder, out_folder, **kwargs)
    try:
        plot_and_gifify_archives(experiment_folder, out_folder, **kwargs)
    except NotImplementedError:
        pass  # function must not be 2d...


def main(**kwargs):
    for experiment_folder in data_path.iterdir():
        print(experiment_folder.name)
        perform_processing_for(experiment_folder, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exts", action="extend", nargs="+", type=str)
    args = parser.parse_args()
    kwargs = {}
    if args.exts:
        kwargs['save_exts'] = args.exts
    main(**kwargs)
