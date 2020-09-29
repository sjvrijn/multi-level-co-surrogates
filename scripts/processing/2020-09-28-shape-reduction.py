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


def extract_right_upper_square(da: xr.DataArray, num_high, num_low) -> xr.DataArray:
    return da.sel(n_high=slice(-num_high, None), n_low=slice(-num_low, None))


def extract_at_interval(da: xr.DataArray, interval) -> xr.DataArray:
    return da.sel(
        n_high=slice(None, None, interval),
        n_low=slice(None, None, interval)
    )


def extract_right_upper_triangle(da: xr.DataArray, num_high, num_low) -> xr.DataArray:
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    square_sizes = [(2*size, 5*size) for size in range(1, 21)]
    intervals = range(1, 11)

    coords = {'square_sizes': square_sizes, 'intervals': intervals}
    dims = list(coords.keys())
    shape = (*[len(coord) for coord in coords.values()], -1)

    for file in filter(lambda x: '.nc' in x.name, kriging_path.iterdir()):
        orig_da = xr.open_dataset(file)['mses'].load()

        results = []
        for square_size, interval in product(square_sizes, intervals):
            smaller_da = extract_right_upper_square(orig_da, 10, 20)
            smaller_da = extract_at_interval(smaller_da, interval)

            results.append(proc.calc_angle(smaller_da))

        ds = xr.Dataset(
            {
                field: (dims, data)
                for field, data in zip(results[0]._fields, np.array(results).reshape(shape))
            },
            coords=coords
        )
        #TODO: do plotting
