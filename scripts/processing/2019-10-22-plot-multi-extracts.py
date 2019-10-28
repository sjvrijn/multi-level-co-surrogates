#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-09-30-plot-wedges.py: Plot the wedge-shaped MSE-grids
for various parametrizations of the Adjustable Branin
function.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import csv
from collections import namedtuple
from pprint import pprint

import numpy as np
import xarray as xr
from pyprojroot import here

import processing as proc

experiment_name = "2019-09-mse-r2"

data_dir = here("files") / experiment_name
plot_dir = here("TMP") / experiment_name
plot_dir.mkdir(parents=True, exist_ok=True)

plot_extension = "png"

Case = namedtuple('Case', 'name ndim vmin vmax max_diff')

cases = [
    Case(f"AdjustableBranin{a1}", 2, None, None, None)
    for a1 in np.round(np.linspace(-0.4, 1.0, 15), 1)
    if a1 != 0.5
]


data_arrays = xr.open_dataset(


for c in cases:
    fname = data_dir / f'Matern_{c.ndim}d_{c.name}.nc'
    if not fname.exists():
        print(f"Could not find {fname}! Skipping...")
        continue

    with xr.open_dataset(fname) as ds:
        mses = ds['mses'].load()

    print(mses.coords)
    print('median')
    pprint([(f'{95 + i}%-ile', np.nanpercentile(mses.median(dim='rep'), 95 + i)) for i in range(6)])

    plot_name = f'{c.ndim}d-{c.name}-high-low-samples-linear'
    title = f'{c.name} ({c.ndim}D)'

    proc.plot_high_vs_low_num_samples(mses, title, vmin=c.vmin, vmax=c.vmax,
                                      save_as=plot_dir / f'{plot_name}.{plot_extension}')
    proc.plot_high_vs_low_num_samples_diff(mses, title, max_diff=c.max_diff,
                                           save_as=plot_dir / f'{plot_name}-diff.{plot_extension}')

    proc.plot_t_scores(mses, title=title,
                       save_as=plot_dir / f'{plot_name}-significance.{plot_extension}')
    proc.plot_extracts(mses, title, save_as=plot_dir / f'{plot_name}-extracts.{plot_extension}', show=True)