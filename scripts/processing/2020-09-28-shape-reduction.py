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


def extract_at_interval(da: xr.DataArray, num_high_interval, num_low_interval) -> xr.DataArray:
    return da.sel(
        n_high=slice(None, None, num_high_interval),
        n_low=slice(None, None, num_low_interval)
    )


def extract_right_upper_triangle(da: xr.DataArray, num_high, num_low) -> xr.DataArray:
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

