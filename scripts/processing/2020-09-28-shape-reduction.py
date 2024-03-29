#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-09-28-shape-reduction.py: Investigating if only taking part of a fully
enumerated Error Grid can still result in similar gradients, i.e.: what is the
minimum Error Grid data to compute?.
"""

import argparse
from collections import namedtuple
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pathlib import Path
from pyprojroot import here

import multiLevelCoSurrogates as mlcs
import processing as proc

print(f'Running script: {__file__}')


kriging_path = here('files/2019-09-mse-nc/')
adjustables_path = here('files/2019-10-07-adjustables')
output_path = here('files/2020-09-28-shape-reduction/', warn=False)
output_path.mkdir(exist_ok=True, parents=True)
PLOT_PATH = here('plots/2020-09-28-shape-reduction/', warn=False)
PLOT_PATH.mkdir(exist_ok=True, parents=True)


def extract_right_upper_rectangle(da: xr.DataArray, num_high, num_low) -> xr.DataArray:
    """Reduce size of of an Error Grid by selecting a rectangle of data
    from the top-right
    """
    return da.where(
        da.n_high.isin(da.n_high[slice(-num_high, None)]) *
        da.n_low.isin(da.n_low[slice(-num_low, None)])
    )


def extract_at_interval(da: xr.DataArray, interval) -> xr.DataArray:
    """Reduce size of an Error Grid by selecting data at a fixed interval along
    both the number of high- and low-fidelity samples.
    """
    return da.where(
        da.n_high.isin(da.n_high[slice(None, None, interval)]) *
        da.n_low.isin(da.n_low[slice(None, None, interval)])
    )


def extract_right_upper_triangle(da: xr.DataArray, num_high, num_low) -> xr.DataArray:
    raise NotImplementedError


def reduce_size(da: xr.DataArray, rect_size, interval) -> xr.DataArray:
    """Collection function to perform all relevant size reductions based
    on the given arguments.
    """
    NHIGH_RECT_STEP = 2
    NLOW_RECT_STEP = 5

    rect_shape = (rect_size * NHIGH_RECT_STEP, rect_size * NLOW_RECT_STEP)
    reduced_size_da = extract_right_upper_rectangle(da, *rect_shape)
    reduced_size_da = extract_at_interval(reduced_size_da, interval)

    return reduced_size_da


def calculate_gradients_of_reduced(da: xr.DataArray, reduction_options: dict) -> xr.Dataset:
    """Calculate the gradients after all specified reductions for an Error Grid
    stored at the given path.
    """

    results = [
        mlcs.utils.error_grids.calc_angle(reduce_size(da, *options))
        for options in product(*reduction_options.values())
    ]

    shape = (-1, *[len(coord) for coord in reduction_options.values()])
    all_data = np.array(results).T.reshape(shape)
    dims = tuple(reduction_options.keys())
    ds_data = {
        field: (dims, data)
        for field, data in zip(results[0]._fields, all_data)
    }
    return xr.Dataset(ds_data, coords=reduction_options)


def get_reduced_gradient_summary(filename: Path, reduction_options: dict, *,
                                 regenerate: bool=False) -> xr.Dataset:
    """Return gradients for reduced-size Error Grids of given file. Calculate
    and store in .nc file if needed, loads from pre-calculated file otherwise.
    """
    summary_filename = output_path / f'gradient-summary-{filename.name}'
    if regenerate or not summary_filename.exists():
        da = xr.open_dataset(filename)['mses'].load().sel(model='high_hier')
        gradients = calculate_gradients_of_reduced(da, reduction_options)
        gradients.to_netcdf(summary_filename)
    else:
        gradients = xr.load_dataset(summary_filename)

    return gradients


def plot_gradients_of_reduced(gradient_summary, case_name):
    line_at_90 = dict(y=90, alpha=.5, color='black')
    grid_style = dict(alpha=.5, linestyle=':')

    fig, ax = plt.subplots(constrained_layout=True)
    for interval in reversed(gradient_summary.coords['intervals'].values):
        sub_ds = gradient_summary.sel(intervals=interval)
        x = sub_ds.coords['rect_sizes'].values
        y = sub_ds['deg']
        errors = np.stack([
            y - sub_ds['deg_low'],
            sub_ds['deg_high'] - y,
        ])
        ax.errorbar(x=x, y=y, yerr=errors, label=f'interval {interval}', capsize=2)
    ax.grid(**grid_style)
    ax.axhline(**line_at_90)
    ax.set_title(case_name)
    ax.set_xlabel('rectangle size *(2, 5)')
    ax.set_ylabel('gradient angle')
    ax.set_ylim([0, 135])
    ax.legend(loc=0)
    for suffix in proc.suffixes:
        fig.savefig(PLOT_PATH / f'gradient-summary-{case_name}{suffix}', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-regen', action='store_true')
    parser.add_argument('--plot-examples')
    args = parser.parse_args()

    if args.plot_examples:
        path = Path(args.plot_examples)
        da = xr.open_dataset(path)['mses'].load()
        Reduction = namedtuple('Reduction', 'rect_size interval name')
        example_reductions = [
            Reduction(20, 1, 'Large Rectangle'),
            Reduction(10, 1, 'Small Rectangle'),
            Reduction(25, 2, 'High Interval'),
            Reduction(25, 5, 'Low Interval'),
            Reduction(20, 5, 'Large Rectangle, low Interval'),
            Reduction(10, 2, 'Small Rectangle, High Interval'),
        ]
        for *reductions, name in example_reductions:
            smaller_da = reduce_size(da, *reductions)

            save_path = PLOT_PATH / f'example-smaller-{name.replace(" ", "-").replace(",", "").lower()}-{path.stem}'
            smaller_da.sel(model='high_hier').median(dim='rep').plot()
            plt.title(f'Example: {name} -- {path.stem}')
            for suffix in proc.suffixes:
                plt.savefig(f'{save_path}{suffix}', dpi=300)
            plt.close()

    all_reductions = dict(
        rect_sizes=range(1, 26),
        intervals=range(1, 11),
    )

    for file in filter(lambda x: '.nc' in x.name, kriging_path.iterdir()):
        ds = get_reduced_gradient_summary(file, all_reductions, regenerate=args.force_regen)
        plot_gradients_of_reduced(ds, case_name=file.stem)

    for file in filter(lambda x: '.nc' in x.name, adjustables_path.iterdir()):
        ds = get_reduced_gradient_summary(file, all_reductions, regenerate=args.force_regen)
        plot_gradients_of_reduced(ds, case_name=file.stem)
