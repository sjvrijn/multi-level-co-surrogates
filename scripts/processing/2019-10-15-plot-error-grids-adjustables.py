#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-10-15-plot-error-grids-adjustables.py: Plot the wedge-shaped
MSE-grids for various parametrizations of the Adjustable
Paciorek, Hartmann3 and Trid functions.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from collections import namedtuple

import numpy as np
import xarray as xr
from pyprojroot import here

import processing as proc

print(f'Running script: {__file__}')

experiment_name = "2019-10-07-adjustables"

data_dir = here("files") / experiment_name
plot_dir = here("plots", warn=False) / experiment_name
plot_dir.mkdir(parents=True, exist_ok=True)

Case = namedtuple('Case', 'name ndim vmin vmax max_diff')

cases = [
    Case(f"Adjustable-{name}-{a1:<04}", ndim, vmin, vmax, None)
    for a1 in np.round(np.linspace(0.0, 1.0, 21), 2)
    for name, ndim, vmin, vmax in (
        ("Branin",     2,   10, 1e4),
        ("Paciorek",   2, 1e-2, 1e2),
        ("Hartmann3",  3, 5e-2, 5e0),
        ("Trid",      10,  5e7, 5e9),
    )
]


for c in cases:
    fname = data_dir / f'Matern-{c.ndim}d-{c.name}.nc'
    if not fname.exists():
        print(f"Could not find {fname}! Skipping...")
        continue

    with xr.open_dataset(fname) as ds:
        mses = ds['mses'].load()

    plot_name = f'{c.ndim}d-{c.name.replace(".","")}-high-low-samples-linear'
    title = f'{c.ndim}D {c.name}'

    common_options = dict(
        data=mses,
        title=title,
        title_width=32,
        vmin=c.vmin,
        vmax=c.vmax,
        contours=8,
        as_log=True,
        gradient_arrow=True,
    )
    try:
        proc.plot_error_grid(**common_options, save_as=plot_dir / f'{plot_name}')
        proc.plot_error_grid(**common_options, save_as=plot_dir / f'no-bar-{plot_name}',
                             include_colorbar=False)
        proc.plot_error_grid(**common_options, save_as=plot_dir / f'no-y-{plot_name}',
                             label_y=False)
        proc.plot_error_grid(**common_options, save_as=plot_dir / f'clean-{plot_name}',
                             include_colorbar=False, label_y=False)
    except ValueError:
        print(f'ValueError encountered for {fname}, continueing...')

    #proc.plot_error_grid_diff(mses, title, max_diff=c.max_diff,
    #                          save_as=plot_dir / f'{plot_name}-diff')

    #proc.plot_t_scores(mses, title=title,
    #                   save_as=plot_dir / f'{plot_name}-significance')
    #proc.plot_extracts(mses, title,
    #                   save_as=plot_dir / f'{plot_name}-extracts')
    #proc.plot_extracts(mses, title, normalize=True,
    #                   save_as=plot_dir / f'{plot_name}-normalized-extracts')
