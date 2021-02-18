#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-11-25-subsampling.py: Plot the error grids
for all available (standard) cases, based on the subsampling
procedure
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from collections import namedtuple
from itertools import product

import xarray as xr
from pyprojroot import here

import processing as proc

print(f'Running script: {__file__}')

experiment_name = "2019-11-15-subsampling"

data_dir = here("files") / experiment_name
plot_dir = here("plots", warn=False) / experiment_name
plot_dir.mkdir(parents=True, exist_ok=True)


Case = namedtuple('Case', 'name ndim vmin vmax max_diff')
Subsample = namedtuple('Subsample', 'high low')

cases = [
    Case('Forrester',        1,  1e-5,     1e2,  100),
    Case('Forrester',        2,  None,    None,  100),
    Case('Forrester',        4,  None,    None,   10),
    Case('Forrester',        6,  None,    None,   10),
    Case('Forrester',        8,  None,    None,   10),
    Case('Bohachevsky',      2,  None,    None,  100),
    Case('Booth',            2,  None,    None, 1000),
    Case('Branin',           2,   1e1,     1e4, None),
    Case('Currin',           2,   .01,      10,   50),
    Case('Himmelblau',       2,  None,    None, 1000),
    Case('Six-Hump-Camelback', 2,  None,    None,  100),
    Case('Park-91A',          4,  1e-3,     1e2,    1),
    Case('Park-91B',          4,  None,    None,    1),
    Case('Hartmann6',        6,  1e-2,    1e-1, None),
    Case('Borehole',         8,    10,    3000,  1e4),
]


surr_names = [
    'Matern',
    # 'ElasticNet',
]

# with open(data_dir/"DoE-configs.csv", newline="") as infile:
#     reader = csv.reader(infile)
#     Data = namedtuple("Data", next(reader))
#     data = list(map(Data._make, reader))

sub_from = [
    Subsample(50, 125),
]

for case, surr_name, sub in product(cases, surr_names, sub_from):
    fname = data_dir / f'{surr_name}-{case.ndim}d-{case.name}-sub{sub.high}-{sub.low}.nc'

    if not fname.exists():
        print(f"{fname} not found, skipping...")
        continue

    with xr.open_dataset(fname) as ds:
        mses = ds['mses'].load()

    plot_name = f'{surr_name}-{case.ndim}d-{case.name}-sub{sub.high}-{sub.low}-high-low-samples-linear'
    title = f'Subsampling {surr_name} {case.name} ({case.ndim}D) from ({sub.high}, {sub.low})'

    # data_points = [d for d in data
    #                if d.function == c.name and int(d.D) == c.ndim]
    # pprint(data_points)

    proc.plot_error_grid(mses, title, vmin=case.vmin, vmax=case.vmax,
                         # points=data_points,
                         save_as=plot_dir / f'{plot_name}',
                         show=False)
    # proc.plot_error_grid_diff(mses, title, max_diff=case.max_diff,
    #                           save_as=plot_dir / f'{plot_name}-diff')
    #
    # proc.plot_t_scores(mses, title=title,
    #                    save_as=plot_dir / f'{plot_name}-significance')
    # proc.plot_extracts(mses, title,
    #                    save_as=plot_dir / f'{plot_name}-extracts', show=True)
    # proc.plot_extracts(mses, title, normalize=True,
    #                    save_as=plot_dir / f'{plot_name}-normalized-extracts', show=True)
