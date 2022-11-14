#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2022-06-15-polar-plot-gradients.py: script to plot the gradient of a range of
error grids as polar coordinates (angle, budget_used) for an optimization run.
"""

import argparse
from itertools import groupby
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
from parse import compile
from pyprojroot import here
from tqdm import tqdm
import xarray as xr

import multiLevelCoSurrogates as mlcs

import processing as proc

data_path = here('files/2020-11-05-simple-mfbo/')
plot_path = here('plots/2022-06-15-polar-plot-gradients/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)

subfolder_template = compile('{func_name}-{method}-c{cost_ratio:f}-b{init_budget:d}-i{idx:d}')
errorgrid_template = compile('errorgrid_{iteration:d}.nc')


def plot_folder_angles_as_polar(folder: Path, exts, force_regen=False):
    if not subfolder_template.parse(folder.name):
        return
    angles, median_angles, budgets = get_budget_and_angles(folder, force_regen=force_regen)
    if not angles:
        return  # no .nc files were present

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
    ax.plot(angles, budgets)
    for ext in exts:
        fig.savefig(plot_path / f'{folder.name}{ext}')
    fig.clear()
    plt.close('all')


def plot_grouped_folder_angles_as_polar(folders, group_name, exts, force_regen=False):
    print(f'plotting group {group_name}')

    fig, axes = plt.subplots(nrows=1, ncols=2, subplot_kw={'projection': 'polar'})
    fig.suptitle(group_name)

    for idx, folder in tqdm(enumerate(list(folders)), leave=False, desc='Experiment reps'):
        if not subfolder_template.parse(folder.name):
            return
        angles, median_angles, budgets = get_budget_and_angles(folder, force_regen=force_regen)
        if not angles:
            return  # no .nc files were present

        axes[0].plot(angles, budgets, lw=.75, c='black', alpha=.9-(idx*.15))
        axes[1].plot(median_angles, budgets, lw=.75, c='black', alpha=.9-(idx*.15))

    for ax, median_only in zip(axes.flatten(), [False, True]):
        ax.set_thetalim(thetamin=0, thetamax=120)
        ax.set_thetagrids([0, 15, 30, 45, 60, 75, 90, 105, 120])
        ax.set_xlabel('Used budget')
        ax.set_ylabel('Gradient angle')
        ax.set_title(f'{"Median-only" if median_only else "All repetitions"}')

    for ext in exts:
        fig.savefig(plot_path / f'{group_name.replace(".", "").replace(" ", "_")}{ext}', bbox_inches='tight')
    fig.clear()
    plt.close('all')


def get_budget_and_angles(folder: Path, force_regen: bool=False):
    angles_filename = folder / 'angles.csv'
    if force_regen or not angles_filename.exists():
        df = calculate_angles(folder)
        df.to_csv(angles_filename, index=False)
    else:
        df = pd.read_csv(angles_filename)

    return [df[c].values.tolist() for c in ['theta', 'median_theta', 'budgets']]


def calculate_angles(folder):
    budgets = [0]
    init_budget = subfolder_template.parse(folder.name)['init_budget']
    df = pd.read_csv(folder / 'log.csv', index_col=0, sep=';')
    budgets.extend((init_budget - df['budget'].values).tolist())

    angles = []
    median_angles = []
    for file in tqdm(sorted(folder.iterdir()), leave=False, desc='EG files'):
        if not errorgrid_template.parse(file.name):
            continue
        with xr.open_dataset(file) as ds:
            da = ds['mses'].sel(model='high_hier')
        with da.load() as da:
            angle_summary = mlcs.utils.error_grids.calc_angle(da)
            median_summary = mlcs.utils.error_grids.calc_angle(da.median(dim='rep'))
        angles.append(angle_summary)
        median_angles.append(median_summary.theta)

    # todo: better error handling on why this would happen and how to deal with it
    if len(angles) != len(budgets):
        actual_length = min(len(angles), len(budgets))
        angles = angles[:actual_length]
        median_angles = median_angles[:actual_length]
        budgets = budgets[:actual_length]

    df = pd.DataFrame.from_records(angles, columns=mlcs.utils.error_grids.AngleSummary._fields)
    df['median_theta'] = median_angles
    df['budgets'] = budgets

    return df


def remove_idx(name: Union[Path, str]) -> str:
    if isinstance(name, Path):
        name = name.name
    return name[:name.find('-i')]


def main(args):
    suffixes = args.exts or proc.suffixes
    folders = []
    for subfolder in sorted(data_path.iterdir()):
        if not subfolder.is_dir():
            continue
        folders.append(subfolder)
        if args.singles:
            plot_folder_angles_as_polar(subfolder, suffixes, force_regen=args.force_regen)

    if args.grouped:
        for name, folder_group in groupby(folders, key=remove_idx):
            plot_grouped_folder_angles_as_polar(folder_group, name, suffixes, force_regen=args.force_regen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exts", action="extend", nargs="+", type=str,
                        help="File extensions to use when saving images. Default: [.PNG, .PDF].")
    parser.add_argument("--singles", action=argparse.BooleanOptionalAction, default=False,
                        help="Plot every run comparison individually. Default: --no-singles.")
    parser.add_argument("--grouped", action=argparse.BooleanOptionalAction, default=True,
                        help="Plot comparison of methods over multiple runs. Default: --grouped.")
    parser.add_argument('--force-regen', action='store_true',
                        help="Force regeneration of all caching files")
    args = parser.parse_args()

    main(args)
