#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2022-07-07-vertical-linreg-distribution.py: script to plot the distribution of
the linear regression fit's vertical component confidence interval on error grids.

Reason is the large number of gradient angles that 'flip' between ~90 and ~270
degrees during the optimization progress
"""

import argparse
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from parse import compile
from pyprojroot import here
from tqdm import tqdm
import xarray as xr

import multiLevelCoSurrogates as mlcs

import processing as proc


data_path = here('files/2020-11-05-simple-mfbo/')
plot_path = here('plots/2022-07-07-vertical-linreg-distribution/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)

subfolder_template = compile('{func_name}-{method}-b{init_budget:d}-i{idx:d}')
errorgrid_template = compile('errorgrid_{iteration:d}.nc')



def plot_vertical_linreg_component(folders, exts, force_regen=False):

    alpha_min, alpha, alpha_max = [], [], []

    for folder in folders:

        df = get_budget_and_angles(folder, force_regen=force_regen)
        if 'SSE' not in df.columns:
            continue

        s = np.sqrt(df['SSE'] / df['df'])
        se_nhigh = s / df['var_nhigh']

        alpha_min.extend((df['alpha'] - 1.96*se_nhigh).values.tolist())
        alpha.extend(df['alpha'].values.tolist())
        alpha_max.extend((df['alpha'] + 1.96*se_nhigh).values.tolist())


    alpha_stuff = list(zip(alpha_min, alpha, alpha_max))
    alpha_stuff.sort(key=itemgetter(1))
    alpha_min, alpha, alpha_max = zip(*alpha_stuff)

    fig, ax = plt.subplots(1, 1)

    ax.plot(alpha, label='vertical component')
    ax.fill_between(np.arange(len(alpha)), alpha_min, alpha_max, color='black', alpha=.3)
    ax.axhline(y=0, color='black')

    ax.set_title('sorted values for the vertical component')

    for ext in exts:
        fig.savefig(plot_path / f'vertical-component{ext}', bbox_inches='tight')
    fig.clear()
    plt.close('all')



def scatter_component_vs_errorgrid_size(folders, exts, force_regen=False):

    alpha = []
    eg_size_high, eg_size_low = [], []
    for folder in folders:

        df = get_budget_and_angles(folder, force_regen=force_regen)
        if 'alpha' not in df.columns:
            continue
        alpha.extend(df['alpha'].values.tolist())

        for file in sorted(folder.iterdir()):
            if file.suffix != '.nc':
                continue
            with xr.open_dataset(file) as ds:
                eg_size_high.append(len(ds.coords['n_high']))
                eg_size_low.append(len(ds.coords['n_low']))


    fig, ax = plt.subplots(1, 1)

    random_offsets = np.random.random(len(eg_size_high)) / 2 - .25

    ax.scatter(alpha, eg_size_low + random_offsets, s=4, alpha=.1, label='low')
    ax.scatter(alpha, eg_size_high + random_offsets, s=4, alpha=.1, label='high')
    ax.axvline(x=0, color='black', linewidth=.5)
    ax.legend(loc=0)
    for ext in exts:
        fig.savefig(plot_path / f'vert-component-scatter{ext}', bbox_inches='tight')
    fig.clear()
    plt.close('all')


def get_budget_and_angles(folder: Path, force_regen: bool=False):
    angles_filename = folder / 'angles.csv'
    if force_regen or not angles_filename.exists():
        df = calculate_angles(folder)
        df.to_csv(angles_filename, index=False)
    else:
        df = pd.read_csv(angles_filename)

    return df


def calculate_angles(folder):
    angles = []
    for file in tqdm(sorted(folder.iterdir()), leave=False, desc='EG files'):
        if not errorgrid_template.parse(file.name):
            continue
        with xr.open_dataset(file) as ds:
            da = ds['mses'].sel(model='high_hier')
        with da.load() as da:
            angle_summary = mlcs.utils.error_grids.calc_angle(da)
        angles.append(angle_summary)

    return pd.DataFrame.from_records(angles, columns=mlcs.utils.error_grids.AngleSummary._fields)



def main(args):
    suffixes = args.exts or proc.suffixes
    folders = [
        subfolder
        for subfolder in sorted(data_path.iterdir())
        if subfolder.is_dir()
    ]

    plot_vertical_linreg_component(folders, suffixes, force_regen=args.force_regen)
    scatter_component_vs_errorgrid_size(folders, suffixes)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exts", action="extend", nargs="+", type=str,
                        help="File extensions to use when saving images. Default: [.PNG, .PDF].")
    parser.add_argument('--force-regen', action='store_true',
                        help="Force regeneration of all caching files")
    args = parser.parse_args()

    main(args)
