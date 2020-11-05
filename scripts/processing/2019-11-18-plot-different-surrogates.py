#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-11-18-different-surrogates.py: Plot the wedge-shaped MSE-grids
for all available (standard) cases, with DoE's as used in
literature plotted.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import csv
from collections import namedtuple
from itertools import product
from pprint import pprint

import xarray as xr
from pyprojroot import here

import processing as proc

experiment_name = "2019-11-12-different-surrogates"

data_dir = here("files") / experiment_name
plot_dir = here("plots") / experiment_name
plot_dir.mkdir(parents=True, exist_ok=True)


Case = namedtuple('Case', 'name ndim vmin vmax max_diff')

cases = [
    Case('Forrester',        1,  None,    None, None),
    Case('Forrester',        2,  None,    None, None),
    Case('Forrester',        4,  None,    None, None),
    # Case('Forrester',        6,  None,    None,   10),
    # Case('Forrester',        8,  None,    None,   10),
    # Case('Bohachevsky',      2,  None,    None,  100),
    # Case('Booth',            2,  None,    None, 1000),
    Case('Branin',           2,  None,    None, None),
    Case('Currin',           2,  None,    None, None),
    # Case('Himmelblau',       2,  None,    None, 1000),
    # Case('Six-Hump-Camelback', 2,  None,    None,  100),
    Case('Park-91A',          4,  None,    None, None),
    # Case('Park-91B',          4,  None,    None,    1),
    Case('Hartmann6',        6,  None,    None, None),
    Case('Borehole',         8,  None,    None, None),
]


surr_names = [
    'ElasticNet',
    'RandomForest',
    'SVM',
]

# with open(data_dir/"DoE-configs.csv", newline="") as infile:
#     reader = csv.reader(infile)
#     Data = namedtuple("Data", next(reader))
#     data = list(map(Data._make, reader))


for case, surr_name in product(cases, surr_names):
    fname = data_dir / f'{surr_name}-{case.ndim}d-{case.name}.nc'

    if not fname.exists():
        print(f"{fname} not found, skipping...")
        continue

    with xr.open_dataset(fname) as ds:
        mses = ds['mses'].load()

    plot_name = f'{surr_name}-{case.ndim}d-{case.name}-high-low-samples-linear'
    title = f'{surr_name} {case.name} ({case.ndim}D)'

    # data_points = [d for d in data
    #                if d.function == c.name and int(d.D) == c.ndim]
    # pprint(data_points)

    proc.plot_error_grid(mses, title, vmin=case.vmin, vmax=case.vmax,
                         # points=data_points,
                         save_as=plot_dir / f'{plot_name}.pdf')
    # proc.plot_error_grid_diff(mses, title, max_diff=case.max_diff,
    #                           save_as=plot_dir / f'{plot_name}-diff.pdf')

    # proc.plot_t_scores(mses, title=title,
    #                    save_as=plot_dir / f'{plot_name}-significance.pdf')
    # proc.plot_extracts(mses, title,
    #                    save_as=plot_dir / f'{plot_name}-extracts.pdf', show=True)
    # proc.plot_extracts(mses, title, normalize=True,
    #                    save_as=plot_dir / f'{plot_name}-normalized-extracts.pdf', show=True)
