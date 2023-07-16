#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2023-07-16-gradients-with-ci.py: script to plot the gradient of a range of
error grids together with the 95% confidence interval.
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
plot_path = here('plots/2023-07-16-gradients-with-ci/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)

subfolder_template = compile('{func_name}-{method}-c{cost_ratio:f}-b{init_budget:d}-i{idx:d}')
errorgrid_template = compile('errorgrid_{iteration:d}.nc')


def plot_folder_angles_as_polar(folder: Path, exts, force_regen=False, use_cost_ratio=None):
    if not subfolder_template.parse(folder.name):
        return
    angles, median_angles, budgets = get_budget_and_angles(folder, force_regen=force_regen, use_cost_ratio=use_cost_ratio)
    if not angles:
        return  # no .nc files were present

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': 'polar'})
    ax.plot(angles, budgets)
    for ext in exts:
        fig.savefig(plot_path / f'{folder.name}{ext}')
    fig.clear()
    plt.close('all')


def plot_grouped_folder_angles_as_polar(folders, group_name, exts, force_regen=False, use_cost_ratio=None, save_median=False):
    print(f'plotting group {group_name}')

    #fig_all, ax_all = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': 'polar'})
    #fig_med, ax_med = plt.subplots(nrows=1, ncols=1, subplot_kw={'projection': 'polar'})
    fig_ci, ax_ci = plt.subplots(nrows=1, ncols=1)
    #fig_all.suptitle(group_name)
    #fig_med.suptitle(group_name)
    fig_ci.suptitle(group_name)

    for idx, folder in tqdm(enumerate(list(folders)), leave=False, desc='Experiment reps'):
        if not subfolder_template.parse(folder.name):
            return
        angles, median_angles, budgets, deg, low, high = get_budget_and_angles(folder, force_regen=force_regen, use_cost_ratio=use_cost_ratio)
        if not angles:
            return  # no .nc files were present

        #ax_ci.plot(budgets, low, c='blue')
        ax_ci.plot(budgets, deg, c='black')
        ax_ci.fill_between(budgets, low, high, alpha=.15, color='black')

        #ax_ci.plot(budgets, high, c='orange')

        #ax_all.plot(angles, budgets, lw=.75, c='black', alpha=.9-(idx*.15))
        #if save_median:
        #    ax_med.plot(median_angles, budgets, lw=.75, c='black', alpha=.9-(idx*.15))

    ax_ci.set_ylim([-5, 125])
    ax_ci.set_xlim([0, max(budgets)])
    ax_ci.hlines([0, 90], 0, max(budgets), color='black', linewidth=.25)
    ax_ci.set_xlabel('Used Budget')
    ax_ci.set_ylabel('Gradient angle')
    #for ax, title in zip([ax_all, ax_med], ["Using all repetitions", "Using only median of repetitions"]):
    #    ax.set_thetalim(thetamin=0, thetamax=120)
    #    ax.set_thetagrids([0, 15, 30, 45, 60, 75, 90, 105, 120])
    #    ax.set_xlabel('Used budget')
    #    ax.set_ylabel('Gradient angle')
    #    ax.set_title(title)

    for ext in exts:
        fig_ci.savefig(plot_path / f'ci_{group_name.replace(".", "").replace(" ", "_")}{ext}', bbox_inches='tight')
        #fig_all.savefig(plot_path / f'{group_name.replace(".", "").replace(" ", "_")}{ext}', bbox_inches='tight')
        #if save_median:
        #    fig_med.savefig(plot_path / f'medians_{group_name.replace(".", "").replace(" ", "_")}{ext}', bbox_inches='tight')
    fig_ci.clear()
    #fig_all.clear()
    #fig_med.clear()
    plt.close('all')


def get_budget_and_angles(folder: Path, force_regen: bool=False, use_cost_ratio: bool=False):
    angles_filename = folder / 'angles.csv'
    if force_regen or not angles_filename.exists():
        df = calculate_angles(folder, use_cost_ratio=use_cost_ratio)
        df.to_csv(angles_filename, index=False)
    else:
        df = pd.read_csv(angles_filename)

    return [df[c].values.tolist() for c in ['theta', 'median_theta', 'budgets', 'deg', 'deg_low', 'deg_high']]


def calculate_angles(folder: Path, use_cost_ratio: bool=False):
    budgets = [0]
    match = subfolder_template.parse(folder.name)
    init_budget = match['init_budget']
    cost_ratio = match['cost_ratio'] if use_cost_ratio else None
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
            angle_summary = mlcs.utils.error_grids.calc_angle(da, cost_ratio=cost_ratio)
            median_summary = mlcs.utils.error_grids.calc_angle(da.median(dim='rep'), cost_ratio=cost_ratio)
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
            plot_folder_angles_as_polar(subfolder, suffixes,
                                        force_regen=args.force_regen,
                                        use_cost_ratio=args.cost_ratio)

    if args.grouped:
        for name, folder_group in groupby(folders, key=remove_idx):
            plot_grouped_folder_angles_as_polar(folder_group, name, suffixes,
                                                force_regen=args.force_regen,
                                                use_cost_ratio=args.cost_ratio,
                                                save_median=args.save_median)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exts", action="extend", nargs="+", type=str,
                        help="File extensions to use when saving images. Default: [.PNG, .PDF].")
    parser.add_argument("--singles", action=argparse.BooleanOptionalAction, default=False,
                        help="Plot every run comparison individually. Default: --no-singles.")
    parser.add_argument("--grouped", action=argparse.BooleanOptionalAction, default=True,
                        help="Plot comparison of methods over multiple runs. Default: --grouped.")
    parser.add_argument('--force-regen', action='store_true',
                        help="Force regeneration of all caching files. Default: --no-force-regen")
    parser.add_argument('--cost-ratio', action='store_true', default=True,
                        help="Include cost-ratio in calculating angles. Default: --cost-ratio")
    parser.add_argument('--save-median', action='store_true',
                        help="Also save grouped plots for only median repetition. Default: --no-save-median")
    args = parser.parse_args()

    main(args)
