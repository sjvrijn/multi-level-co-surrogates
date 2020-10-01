#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-09-28-shape-reduction.py: Investigating if only taking part of a fully
enumerated Error Grid can still result in similar gradients, i.e.: what is the
minimum Error Grid data to compute?.
"""

import argparse
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from pyprojroot import here

import processing as proc


kriging_path = here('files/2019-09-mse-nc/')
output_path = here('files/2020-09-28-shape-reduction/', warn=False)
output_path.mkdir(exist_ok=True, parents=True)
plot_path = here('plots/2020-09-28-shape-reduction/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)


def extract_right_upper_square(da: xr.DataArray, num_high, num_low) -> xr.DataArray:
    """Reduce size of of an Error Grid by selecting a rectangle of data
    from the top-right
    """
    return da.sel(n_high=slice(-num_high, None), n_low=slice(-num_low, None))


def extract_at_interval(da: xr.DataArray, interval) -> xr.DataArray:
    """Reduce size of an Error Grid by selecting data at a fixed interval along
    both the number of high- and low-fidelity samples.
    """
    return da.sel(
        n_high=slice(None, None, interval),
        n_low=slice(None, None, interval)
    )


def extract_right_upper_triangle(da: xr.DataArray, num_high, num_low) -> xr.DataArray:
    raise NotImplementedError


def reduce_size(da: xr.DataArray, square_size, interval) -> xr.DataArray:
    """Collection function to perform all relevant size reductions based
    on the given arguments.
    """
    reduced_size_da = extract_right_upper_square(da, 2 * square_size, 5 * square_size)
    reduced_size_da = extract_at_interval(reduced_size_da, interval)
    return reduced_size_da


def get_reduced_gradient_summary(filename: Path, reduction_options: dict, *,
                                 regenerate: bool=False) -> xr.Dataset:
    """Return gradients for reduced-size Error Grids of given file. Calculate
    and store in .nc file if needed, loads from pre-calculated file otherwise.
    """
    if regenerate or not filename.exists():
        gradients = calculate_gradients_of_reduced(filename, reduction_options)
        gradients.to_netcdf(output_path / f'gradient-summary-{filename.name}')
    else:
        gradients = xr.load_dataset(filename)

    return gradients


def calculate_gradients_of_reduced(filename, reduction_options):
    """"""

    dims = tuple(reduction_options.keys())
    shape = (*[len(coord) for coord in reversed(reduction_options.values())], -1)

    orig_da = xr.open_dataset(filename)['mses'].load().sel(model='high_hier')
    results = [
        proc.calc_angle(reduce_size(orig_da, *options))
        for options in product(reduction_options.values())
    ]
    all_data = np.array(results).reshape(shape).T
    ds_data = {
        field: (dims, data)
        for field, data in zip(results[0]._fields, all_data)
    }
    return xr.Dataset(ds_data, coords=reduction_options)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regen-gradients', action='store_true')
    args = parser.parse_args()

    reductions = dict(
        intervals=range(1, 11),
        square_sizes=range(1, 21),
    )

    for file in filter(lambda x: '.nc' in x.name, kriging_path.iterdir()):
        ds = get_reduced_gradient_summary(file, reductions, regenerate=args.regen_gradients)

        #TODO: do plotting
