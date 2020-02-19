#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-02-18-combined-wedges.py: A processing script file to create both the
regular wedge plots and those for the subsampling results at the same time.
By combining the renders, they are guaranteed to use the same colorscale.
"""

from collections import namedtuple
from itertools import product

import xarray as xr
from pyprojroot import here

import processing as proc

source_regular = "2019-09-mse-nc"
source_subsample = "2019-11-15-subsampling"

output_name = '2020-02-18-combined-wedges/'

regular_dir = here("files") / source_regular
subsample_dir = here("files") / source_subsample
plot_dir = here("plots") / output_name
plot_dir.mkdir(parents=True, exist_ok=True)

Case = namedtuple('Case', 'name ndim as_log')
Subsample = namedtuple('Subsample', 'high low')

cases = [
    Case('Forrester',        1, True),
    Case('Forrester',        2, True),
    Case('Forrester',        4, True),
    Case('Forrester',        6, True),
    Case('Forrester',        8, True),
    Case('Bohachevsky',      2, True),
    Case('Booth',            2, True),
    Case('Branin',           2, True),
    Case('Currin',           2, True),
    Case('Himmelblau',       2, True),
    Case('SixHumpCamelback', 2, True),
    Case('Park91A',          4, True),
    Case('Park91B',          4, True),
    Case('Hartmann6',        6, True),
    Case('Borehole',         8, True),
]

surr_names = [
    'Matern',
    # 'ElasticNet',
]

sub_from = [
    Subsample(50, 125),
]


for case, surr_name, sub in product(cases, surr_names, sub_from):
    subsample_fname = subsample_dir / f'{surr_name}-{case.ndim}d-{case.name}-sub{sub.high}-{sub.low}.nc'
    regular_fname = regular_dir / f'Matern-{case.ndim}d-{case.name}.nc'

    if not subsample_fname.exists():
        print(f"{subsample_fname.name} not found, skipping...")
        continue
    if not regular_fname.exists():
        print(f"{regular_fname.name} not found, skipping...")
        continue

    with xr.open_dataset(subsample_fname) as ds:
        subsample_mses = ds['mses'].load()
    with xr.open_dataset(regular_dir / f'Matern-{case.ndim}d-{case.name}.nc') as ds:
        regular_mses = ds['mses'].load()

    mses = [regular_mses, subsample_mses]
    titles = [
        f'{case.name} ({case.ndim}D)',
        f'Subsampling {surr_name} {case.name} ({case.ndim}D) from ({sub.high}, {sub.low})',
    ]

    plot_name = f'comparison-{surr_name}-{case.ndim}d-{case.name}-sub{sub.high}-{sub.low}-high-low-samples-linear'

    proc.plot_two_high_vs_low_num_samples(mses, titles, case.as_log,
                                          save_as=plot_dir / f'{plot_name}.pdf')

