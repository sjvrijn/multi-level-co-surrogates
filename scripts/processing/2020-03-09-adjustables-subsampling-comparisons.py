#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-03-09-adjustables-subsampling-comparisons.py: A processing script file to create both the
regular wedge plots and those for the subsampling results at the same time.
By combining the renders, they are guaranteed to use the same colorscale.
"""

from collections import namedtuple
from itertools import product

import xarray as xr
from pyprojroot import here

import mf2

import processing as proc

print(f'Running script: {__file__}')

source_regular = "2019-10-07-adjustables"
source_subsample = "2020-03-04-cv-adjustables-subsampling"

output_name = '2020-03-09-adjustables-subsampling-comparisons'

regular_dir = here("files") / source_regular
subsample_dir = here("files") / source_subsample
plot_dir = here("plots", warn=False) / output_name
plot_dir.mkdir(parents=True, exist_ok=True)

Subsample = namedtuple('Subsample', 'high low')

cases = [
    *[mf2.adjustable.branin(a1)    for a1 in [0.00, 0.05, 0.25]],
    *[mf2.adjustable.paciorek(a2)  for a2 in [0.05, 0.10, 0.20, 0.25]],
    *[mf2.adjustable.hartmann3(a3) for a3 in [0.20, 0.25, 0.35, 0.40]],
    *[mf2.adjustable.trid(a4)      for a4 in [0.65, 0.70, 0.80, 0.90, 1.00]],
]

surr_names = [
    'Matern',
]

sub_from = [
    Subsample(50, 125),
]
seed_offsets = range(0, 5)
as_log = True

def format_case_name(name):
    parts = name.split(' ')
    parts[-1] = f'{float(parts[-1]):0.2f}'
    return '-'.join(parts)


for case, surr_name, sub, seed_offset in product(cases, surr_names, sub_from, seed_offsets):
    case_name = format_case_name(case.name)
    subsample_fname = subsample_dir / f'{surr_name}-{case.ndim}d-{case_name}-sub{sub.high}-{sub.low}-seed{seed_offset}.nc'
    regular_fname = regular_dir / f'Matern-{case.ndim}d-{case_name}.nc'

    if not subsample_fname.exists():
        print(f"{subsample_fname.name} not found, skipping...")
        continue
    if not regular_fname.exists():
        print(f"{regular_fname.name} not found, skipping...")
        continue

    with xr.open_dataset(regular_dir / f'Matern-{case.ndim}d-{case_name}.nc') as ds:
        regular_mses = ds['mses'].load()
    with xr.open_dataset(subsample_fname) as ds:
        subsample_mses = ds['mses'].load()
        cv_mses = ds['cv_mses']

    mses = [regular_mses, subsample_mses, cv_mses]
    titles = [
        f'{case.name} ({case.ndim}D)',
        f'Subsampling from {sub.high}, {sub.low}',
        f'Cross-validation of subsampling',
    ]

    plot_name = f'comparison-{surr_name}-{case.ndim}d-{case_name.replace(".","")}-sub{sub.high}-{sub.low}-seed{seed_offset}-high-low-samples'

    proc.plot_multiple_error_grids(mses, titles, as_log, contours=8, gradient_arrow=True,
                                   save_as=plot_dir / f'{plot_name}')

