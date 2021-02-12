#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-09-19-plot-error-grids.py: Plot the wedge-shaped MSE-grids
for all available (standard) cases, with DoE's as used in
literature plotted.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import csv
from collections import namedtuple
from pprint import pprint

import xarray as xr
from pyprojroot import here

import processing as proc

experiment_name = "2019-09-mse-nc"

data_dir = here("files") / experiment_name
plot_dir = here("plots", warn=False) / experiment_name
plot_dir.mkdir(parents=True, exist_ok=True)

Case = namedtuple('Case', 'name ndim vmin vmax max_diff as_log')

cases = [
    Case('Forrester',          1,   1e-5  ,   1e2  ,  100,  True),
    Case('Forrester',          2, 10**-0.8,   1e1  ,  100,  True),
    Case('Forrester',          4,    1e0  ,   1e1  ,   10, False),
    Case('Forrester',          6,    1e0  ,   1e1  ,   10, False),
    Case('Forrester',          8,    1e0  ,   1e1  ,   10, False),
    Case('Bohachevsky',        2, 10**-0.5,   1e3  ,  100,  True),
    Case('Booth',              2,    1e2  ,   1e5  , 1000,  True),
    Case('Branin',             2,    1e1  ,   1e4  , None,  True),
    Case('Currin',             2,   1e-2  ,   1e1  ,   50,  True),
    Case('Himmelblau',         2,    1e2  ,   1e4  , 1000,  True),
    Case('Six-Hump-Camelback', 2,  10**0.5,   1e3  ,  100,  True),
    Case('Park-91A',           4,   1e-3  ,   1e1  ,    1,  True),
    Case('Park-91B',           4,   1e-4  ,   1e0  ,    1,  True),
    Case('Hartmann6',          6,   1e-2  ,  1e-1  , None, False),
    Case('Borehole',           8,    1e1  ,   1e4  ,  1e4,  True),
]

#with open(data_dir/"DoE-configs.csv", newline="") as infile:
#    reader = csv.reader(infile)
#    Data = namedtuple("Data", next(reader))
#    data = list(map(Data._make, reader))
data = []


for c in cases:
    fname = f'Matern-{c.ndim}d-{c.name}.nc'
    try:
        with xr.open_dataset(data_dir / fname) as ds:
            mses = ds['mses'].load()
    except FileNotFoundError:
        print(f"File {fname} not found, skipping ...")
        continue

    plot_name = f'{c.ndim}d-{c.name}-high-low-samples-linear'
    title = f'{c.ndim}D {c.name}'

    data_points = [d for d in data
                   if d.function == c.name and int(d.D) == c.ndim]
    if data_points:
        pprint(data_points)

    proc.plot_error_grid(mses, title, vmin=c.vmin, vmax=c.vmax,
                         points=data_points, contours=8, as_log=c.as_log,
                         save_as=plot_dir / plot_name)
    #proc.plot_error_grid(mses, title, vmin=c.vmin, vmax=c.vmax, contours=8, as_log=True,
    #                     save_as=plot_dir / f'no-bar-{plot_name}', include_colorbar=False)
    proc.plot_error_grid(mses, title, vmin=c.vmin, vmax=c.vmax, contours=8, as_log=True,
                         save_as=plot_dir / f'no-y-{plot_name}', label_y=False)
    #proc.plot_error_grid(mses, title, vmin=c.vmin, vmax=c.vmax, contours=8, as_log=True,
    #                     save_as=plot_dir / f'clean-{plot_name}', include_colorbar=False, label_y=False)

    #proc.plot_error_grid_diff(mses, title, max_diff=c.max_diff,
    #                          save_as=plot_dir / f'{plot_name}-diff')
    #proc.plot_t_scores(mses, title=title,
    #                   save_as=plot_dir / f'{plot_name}-significance')
    #proc.plot_extracts(mses, title,
    #                   save_as=plot_dir / f'{plot_name}-extracts', show=True)
    #proc.plot_extracts(mses, title, normalize=True,
    #                   save_as=plot_dir / f'{plot_name}-normalized-extracts', show=True)
