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

data_dir = here('files/')
plot_dir = here('plots/2019-09_MSE_analysis')
plot_dir.mkdir(parents=True, exist_ok=True)


Case = namedtuple('Case', 'name ndim vmin vmax max_diff')

cases = [
    Case('Forrester', 1, None, None, 100),
    #     Case('Forrester',        2,  None,    None,  100),
    #     Case('Forrester',        4,  None,    None,   10),
    #     Case('Forrester',        6,  None,    None,   10),
    #     Case('Forrester',        8,  None,    None,   10),
    #     Case('Bohachevsky',      2,   500,   2_000,  200),
    #     Case('Booth',            2,   1e5,     5e6, 5000),
    Case('Branin', 2, 10, 1e4, None),
    #     Case('Currin',           2,   .01,      10,   50),
    #     Case('Himmelblau',       2,  None,    None, 1000),
    #     Case('SixHumpCamelBack', 2,  None,    None,  100),
    #     Case('Park91a',          4,  None,    None,    1),
    #     Case('Park91b',          4,  None,    None,    1),
    Case('Hartmann6', 6, 8e-3, 5e-1, 1),
    Case('Borehole', 8, 10, 3000, 1e4),
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

    proc.display_paired_differences(mses, title=title,
                                    save_as=plot_dir / f'{plot_name}_significance.pdf')
    proc.plot_extracts(mses, title, save_as=plot_dir / f'{plot_name}_extracts.pdf', show=True)
