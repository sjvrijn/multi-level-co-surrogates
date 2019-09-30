#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Filename.py: << A short summary docstring about this file >>
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

experiment_name  "2019-09-mse-r2"

data_dir = here("files") / experiment_name
plot_dir = here("plots") / experiment_name
plot_dir.mkdir(parents=True, exist_ok=True)


Case = namedtuple('Case', 'name ndim vmin vmax max_diff')

cases = [
    Case(f"AdjustableBranin{a1}", 2, 10, 1e4, 1000),
    for a1 in np.linspace(-0.5, 1.0, 16)
]


for c in cases:
    with xr.open_dataset(data_dir / f'Matern_{c.ndim}d_{c.name}.nc') as ds:
        mses = ds['mses'].load()

    print(mses.coords)
    print('median')
    pprint([(f'{95 + i}%-ile', np.nanpercentile(mses.median(dim='rep'), 95 + i)) for i in range(6)])

    plot_name = f'{c.ndim}d-{c.name}-high-low-samples-linear'
    title = f'{c.name} ({c.ndim}D)'

    proc.plot_high_vs_low_num_samples(mses, title, vmin=c.vmin, vmax=c.vmax,
                                      save_as=plot_dir / f'{plot_name}.pdf')
    proc.plot_high_vs_low_num_samples_diff(mses, title, max_diff=c.max_diff,
                                           save_as=plot_dir / f'{plot_name}_diff.pdf')

    proc.plot_t_scores(mses, title=title,
                       save_as=plot_dir / f'{plot_name}_significance.pdf')
    proc.plot_extracts(mses, title, save_as=plot_dir / f'{plot_name}_extracts.pdf', show=True)
