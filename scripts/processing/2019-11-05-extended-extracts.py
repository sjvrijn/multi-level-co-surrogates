#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-11-05-extended-extracts.py: experiment script for combining
the extend-ratios data with the original mse-nc data files
to plot relative/normalized extracts up to a consistent high/low ratio
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from collections import namedtuple

import xarray as xr
from pyprojroot import here

import processing as proc

print(f'Running script: {__file__}')

data_dir = here("files")

base_data = data_dir / "2019-09-mse-nc"
extended_data = data_dir / "2019-11-01-extend-ratios"

plot_dir = here("plots", warn=False) / "2019-11-05-extended-extracts"
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
    Case('Branin',           2,  None,    None, None),
    Case('Currin',           2,   .01,      10,   50),
    Case('Himmelblau',       2,  None,    None, 1000),
    Case('Six-Hump-Camelback', 2,  None,    None,  100),
    Case('Park91A',          4,  None,    None,    1),
    Case('Park91B',          4,  None,    None,    1),
    Case('Hartmann6',        6,  None,    None, None),
    Case('Borehole',         8,    10,    3000,  1e4),
]


for c in cases:
    try:
        fname = f'Matern-{c.ndim}d-{c.name}.nc'
        with xr.open_dataset(base_data / fname) as ds:
            base_mses = ds['mses'].load()
        with xr.open_dataset(extended_data / fname) as ds:
            extended_mses = ds['mses'].load()
    except FileNotFoundError:
        print(f'One of {fname} files not found, skipping...')
        continue

    try:
        mses = base_mses.combine_first(extended_mses)
    except Exception as e:
        print(f'Failure to merge base with extension for {c.ndim}d {c.name}')
        print(e)
        continue

    plot_name = f'{c.ndim}d-{c.name}-high-low-samples-linear'
    title = f'{c.name} ({c.ndim}D) - Extended'

    proc.plot_extracts(mses, title,
                       save_as=plot_dir / f'{plot_name}-extended-extracts', show=True)
    proc.plot_extracts(mses, title, normalize=True, max_x=20,
                       save_as=plot_dir / f'{plot_name}-extended-normalized-extracts', show=True)
