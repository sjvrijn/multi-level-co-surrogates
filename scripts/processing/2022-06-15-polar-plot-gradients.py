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
import xarray as xr

import multiLevelCoSurrogates as mlcs

import processing as proc

data_path = here('files/2020-11-05-simple-mfbo/')
plot_path = here('plots/2022-06-15-polar-plot-gradients/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)

subfolder_template = compile('{func_name}-{method}-b{init_budget:d}-i{idx:d}')
errorgrid_template = compile('errorgrid_{iteration:d}.nc')


def plot_folder_angles_as_polar(folder: Path, exts):
    if not subfolder_template.parse(folder.name):
        return
    angles, budgets = load_budget_and_angles(folder)
    if not angles:
        return  # no .nc files were present

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
    ax.plot(angles, budgets)
    for ext in exts:
        fig.savefig(plot_path / f'{folder.name}{ext}')
    fig.clear()
    plt.close('all')


def plot_grouped_folder_angles_as_polar(group_of_folders, group_name, exts):
    print(f'plotting group {group_name}')

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
    for folder in group_of_folders:
        if not subfolder_template.parse(folder.name):
            return
        angles, budgets = load_budget_and_angles(folder)
        if not angles:
            return  # no .nc files were present

        ax.plot(angles, budgets, lw=.75, c='black')

    ax.set_thetalim(thetamin=0, thetamax=120)
    ax.set_thetagrids([0, 15, 30, 45, 60, 75, 90, 105, 120])
    ax.set_xlabel('Used budget')
    ax.set_ylabel('Error Grid gradient angle')
    ax.set_title(group_name)
    for ext in exts:
        fig.savefig(plot_path / f'{group_name}{ext}', bbox_inches='tight')
    fig.clear()
    plt.close('all')


def load_budget_and_angles(folder):
    budgets = [0]
    init_budget = subfolder_template.parse(folder.name)['init_budget']
    df = pd.read_csv(folder / 'log.csv', index_col=0, sep=';')
    budgets.extend((init_budget - df['budget'].values).tolist())

    # todo: rewrite proc.get_gradient_angles to be more generic?
    # angle_df = proc.get_gradient_angles(folder)
    angles = []
    for file in sorted(folder.iterdir()):
        if not errorgrid_template.parse(file.name):
            continue
        with xr.open_dataset(file) as ds:
            da = ds['mses'].sel(model='high_hier')
        with da.load() as da:
            angle_summary = mlcs.utils.error_grids.calc_angle(da)
        angles.append(angle_summary.theta)

    if len(angles) != len(budgets):
        actual_length = min(len(angles), len(budgets))
        angles, budgets = angles[:actual_length], budgets[:actual_length]

    return angles, budgets


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
        # plot_folder_angles_as_polar(subfolder, suffixes)

    for name, folder_group in groupby(folders, key=remove_idx):
        plot_grouped_folder_angles_as_polar(folder_group, name, suffixes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exts", action="extend", nargs="+", type=str,
                        help="File extensions to use when saving images. Default: [.PNG, .PDF].")
    args = parser.parse_args()

    main(args)
