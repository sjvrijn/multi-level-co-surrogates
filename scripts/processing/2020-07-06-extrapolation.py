#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-07-06-extrapolation.py: based on small subsampled DoEs, extrapolate a best
suggested next DoE size and compare this with results from enumerated DoEs.
"""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pyprojroot import here

import processing as proc


__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


regular_dir = here('files/2019-09-mse-nc')
adjustables_dir = here('files/2019-10-07-adjustables')

subsampling_dir = here('files/2020-03-03-cv-subsampling/')
adjustables_subsampling_dir = here("files/2020-03-04-cv-adjustables-subsampling/")

plot_dir = here('plots/2020-07-06-extrapolation/', warn=False)
plot_dir.mkdir(exist_ok=True, parents=True)


def atan_deg(*args):
    return np.rad2deg(np.arctan2(*args))


def angle_from_linreg(reg):
    return atan_deg(*reg.coef_[:2]) % 180


def calc_intercept(b, h0, l0, gradient, costratio):
    l = (b - h0 + gradient*l0) / (gradient + costratio)
    h = b - costratio*l
    return h, l



# Currently Hardcoded:
# small_h, small_l = 30, 75
# large_h, large_l = 50, 125
costratio = 0.4
# new_budget = large_h + costratio * small_l  # or small_h + costratio * large_l


num_lows = xr.DataArray(np.arange(75, 126), dims=["idx"])
num_highs = xr.DataArray(np.floor(np.linspace(50, 30, len(num_lows))), dims=["idx"])
angles = atan_deg(num_highs-30, num_lows-75)


def plot_extrapolation_suggestion(file_small, file_large):

    with xr.open_dataset(file_small) as ds:
        da = ds['mses'].sel(model='high_hier')
    with da.load() as da:
        reg = proc.fit_lin_reg(da)
        gradient = reg.coef_[0] / reg.coef_[1]
        deg_small = angle_from_linreg(reg)

    with xr.open_dataset(file_large) as ds:
        da = ds['mses'].sel(model='high_hier')
    with da.load() as da:
        deg_large = angle_from_linreg(proc.fit_lin_reg(da))
        data_along_budget_line = da.median(dim='rep').sel(n_low=num_lows, n_high=num_highs).values

    print(deg_large, deg_small)

    smallest_at_angle = angles[np.argmin(data_along_budget_line)]
    gradient_budget_intercept = calc_intercept(80, 30, 75, gradient, costratio)

    plt.figure(figsize=(4.8, 2.4), constrained_layout=True)
    plt.plot(angles, data_along_budget_line, marker='o', label='MSEs from DoE enumeration')
    plt.ylabel('$MSE')
    plt.xlabel('angle measured from (30, 75)')
    plt.axvline(deg_small, ls=':', label='Predicted best angle', color='C1')
    plt.legend(loc=0)
    plt.xlim([0,90])
    plt.ylim(bottom=0)
    plt.title(file_large.stem)
    plt.savefig(plot_dir / f'{file_small.stem.replace(".", "")}.pdf', bbox_inches='tight')
    plt.close()





for file_small in filter(lambda f: 'sub30-75' in str(f), subsampling_dir.iterdir()):
    file_large = regular_dir / file_small.name.replace('-sub30-75-seed0', '')
    plot_extrapolation_suggestion(file_small, file_large)

for file_small in filter(lambda f: 'sub30-75' in str(f), adjustables_subsampling_dir.iterdir()):
    file_large = adjustables_dir / file_small.name.replace('-sub30-75-seed0', '')
    plot_extrapolation_suggestion(file_small, file_large)
