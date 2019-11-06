#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-10-22-plot-multi-extracts.py: Plot extracts over all parameter
values for the adjustable MFF's in single plots to show the progression
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from collections import namedtuple

import numpy as np
import xarray as xr
from pyprojroot import here

import processing as proc

experiments = [
    "2019-09-mse-r2", 
    "2019-10-07-adjustables"
]

for experiment_name in experiments:

    data_dir = here("files") / experiment_name
    plot_dir = here("plots") / "2019-10-multi-extracts"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_extension = "pdf"

    Case = namedtuple('Case', 'name ndim vmin vmax max_diff')
    CaseSet = namedtuple('CaseSet', 'generic_name ndim cases')

    case_sets = [
        CaseSet(f'Adjustable{name}', ndim, [
            Case(f"Adjustable{name}{a1}", ndim, None, None, None)
            for a1 in np.round(np.linspace(0.0, 1.0, 21), 2)
        ])
        for name, ndim in (('Branin', 2),
                           ('Paciorek', 2),
                           ('Hartmann3', 3),
                           ('Trid', 10))
    ]


    for case_set in case_sets:
        data_arrays = []
        for case in case_set.cases:
            fname = data_dir / f'Matern-{case.ndim}d-{case.name}.nc'
            if not fname.exists():
                print(f"Could not find {fname}! Skipping...")
                continue
            ds = xr.open_dataset(fname)
            data_arrays.append(ds['mses'])

        plot_name = f'{case_set.ndim}d-{case_set.generic_name}-high-low-samples-linear'
        title = f'{case_set.generic_name} ({case_set.ndim}D)'

        proc.plot_multi_file_extracts(data_arrays, title=title,
                                      save_as=plot_dir / f'{plot_name}-multi-extracts.{plot_extension}')
