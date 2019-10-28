#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-09-19-plot-wedges.py: Plot the wedge-shaped MSE-grids
for all available (standard) cases, with DoE's as used in
literature plotted.
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

experiment_name = "2019-09-mse-nc"

data_dir = here("files") / experiment_name
plot_dir = here("plots") / experiment_name
plot_dir.mkdir(parents=True, exist_ok=True)


Case = namedtuple('Case', 'name ndim vmin vmax max_diff')

cases = [
    Case('Forrester',        1,  None,    None,  100),
    Case('Forrester',        2,  None,    None,  100),
    Case('Forrester',        4,  None,    None,   10),
    Case('Forrester',        6,  None,    None,   10),
    Case('Forrester',        8,  None,    None,   10),
    Case('Bohachevsky',      2,  None,    None,  100),
    Case('Booth',            2,  None,    None, 1000),
    Case('Currin',           2,   .01,      10,   50),
    Case('Himmelblau',       2,  None,    None, 1000),
    Case('SixHumpCamelback', 2,  None,    None,  100),
    Case('Park91A',          4,  None,    None,    1),
    Case('Park91B',          4,  None,    None,    1),
    Case('Hartmann6',        6,  None,    None, None),
    Case('Borehole',         8,    10,    3000,  1e4),
]


with open(data_dir/"DoE_configs.csv", newline="") as infile:
    reader = csv.reader(infile)
    Data = namedtuple("Data", next(reader))
    data = list(map(Data._make, reader))


for c in cases:
    with xr.open_dataset(data_dir / f'Matern_{c.ndim}d_{c.name}.nc') as ds:
        mses = ds['mses'].load()

    print(mses.coords)
    print('median')
    pprint([(f'{95 + i}%-ile', np.nanpercentile(mses.median(dim='rep'), 95 + i)) for i in range(6)])

    plot_name = f'{c.ndim}d-{c.name}-high-low-samples-linear'
    title = f'{c.name} ({c.ndim}D)'

    data_points = [d for d in data
                   if d.function == c.name and int(d.D) == c.ndim]
    pprint(data_points)

    proc.plot_high_vs_low_num_samples(mses, title, vmin=c.vmin, vmax=c.vmax,
                                      points=data_points,
                                      save_as=plot_dir / f'{plot_name}.pdf')
    proc.plot_high_vs_low_num_samples_diff(mses, title, max_diff=c.max_diff,
                                           save_as=plot_dir / f'{plot_name}_diff.pdf')

    proc.plot_t_scores(mses, title=title,
                       save_as=plot_dir / f'{plot_name}_significance.pdf')
    proc.plot_extracts(mses, title, save_as=plot_dir / f'{plot_name}_extracts.pdf', show=True)