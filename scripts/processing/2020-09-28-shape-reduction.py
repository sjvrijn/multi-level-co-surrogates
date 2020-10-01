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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    square_sizes = range(1, 21)
    intervals = range(1, 11)

    coords = {'intervals': list(intervals), 'square_sizes': list(square_sizes)}
    dims = tuple(coords.keys())
    shape = (*[len(coord) for coord in reversed(coords.values())], -1)

    for file in filter(lambda x: '.nc' in x.name, kriging_path.iterdir()):
        print(file)
        orig_da = xr.open_dataset(file)['mses'].load().sel(model='high_hier')

        results = []
        for square_size, interval in product(square_sizes, intervals):
            smaller_da = extract_right_upper_square(orig_da, 2*square_size, 5*square_size)
            smaller_da = extract_at_interval(smaller_da, interval)
            results.append(proc.calc_angle(smaller_da))

        all_data = np.array(results).reshape(shape).T
        ds_data = {
            field: (dims, data)
            for field, data in zip(results[0]._fields, all_data)
        }
        ds = xr.Dataset(ds_data, coords=coords)
        ds.to_netcdf(output_path / f'gradient-summary-{file.name}')

        #TODO: do plotting
