#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
2020-03-09-subsampling-gradients.py: script to create scatterplots of the
gradients in the error for the actual, subsampled, and cross-validated MSE
measures.
Creates a csv of the all relevant values as intermediate step to speed up any
potential rerun.
'''

from collections import namedtuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import xarray as xr
from parse import Parser
from pyprojroot import here
from sklearn.linear_model import LinearRegression

import processing as proc

print(f'Running script: {__file__}')


__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

subsampling_dir = here('files/2020-03-04-cv-adjustables-subsampling')
plot_dir = here('plots/2020-03-09-subsampling-gradients/', warn=False)
plot_dir.mkdir(exist_ok=True, parents=True)

correlations_file = here('files/2019-10-30-correlations/extended_correlations.csv')
gradients_file = here('files/gradient_comparison.csv', warn=False)
fname_parser = Parser('{}-{ndim:d}d-Adjustable-{fname:w}-{param:f}-sub50-125-seed{seed_offset:d}.nc')


def calculate_gradient_comparisons():
    correlations = pd.read_csv(correlations_file)
    Record = namedtuple('Record', 'name param seed_offset pearson_r orig_deg sub_deg cv_deg')
    records = []
    for file in sorted(subsampling_dir.iterdir()):
        match = fname_parser.parse(file.name)
        if not match:
            continue

        func_name, param, seed_offset = match['fname'], match['param'], match['seed_offset']
        orig_row = correlations.loc[(correlations['fname'] == func_name.lower())
                                    & (correlations['param'] == float(param))]
        orig_deg = orig_row['deg'].values[0]
        pearson_r = orig_row['pearson_r'].values[0]

        with xr.open_dataset(file) as ds:
            da = ds['mses'].sel(model='high_hier')
            with da.load() as da:
                reg = proc.fit_lin_reg(da)
            sub_deg = np.rad2deg(np.arctan2(*reg.coef_[:2])) % 180

            da = ds['cv_mses'].sel(model='high_hier')
            with da.load() as da:
                reg = proc.fit_lin_reg(da)
            cv_deg = np.rad2deg(np.arctan2(*reg.coef_[:2])) % 180

        records.append(Record(func_name, param, seed_offset, pearson_r, orig_deg, sub_deg, cv_deg))
    return pd.DataFrame(records)


if gradients_file.exists():
    gradients = pd.read_csv(gradients_file)
else:
    gradients = calculate_gradient_comparisons()
    gradients.to_csv(gradients_file, index=False)


grid_style = dict(b=True, alpha=.5, linestyle=':')
scatter_style = {'s': 12}
diag_line = {'linestyle': '--', 'color': 'black', 'alpha':.3, 'linewidth':.5}
width, height = 7, 3.5


fig, axes = plt.subplots(ncols=2, figsize=(width, height), constrained_layout=True)
for (name, sub_df), marker in zip(gradients.groupby('name'), 'ov^x'):
    axes[0].scatter(sub_df['orig_deg'], sub_df['sub_deg'], label=name, **scatter_style, marker=marker)
    axes[1].scatter(sub_df['orig_deg'], sub_df['cv_deg'], label=name, **scatter_style, marker=marker)

axes[0].set_title('Enumeration angle vs\n subsampled angle')
axes[0].set_ylabel('angle (subsampling)')
axes[0].legend(loc=0)

axes[1].set_title('Enumeration angle vs\n cross-validated angle')
axes[1].set_ylabel('angle (cross-validated subsampling)')

for idx, ax in enumerate(axes):
    ax.set_xlabel('angle (full enumeration)')
    ax.grid(**grid_style)
    ax.plot(np.arange(100), **diag_line)
    ax.set_xlim([0,90])
    ax.set_ylim([0,90])
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))

for ext in proc.extensions:
    fig.savefig(plot_dir / f'scatter_compare.{ext}', dpi=300)
    fig.savefig(plot_dir / f'scatter_compare-sub0.{ext}', dpi=300,
                bbox_inches=Bbox([[0,0],[width/2,height]]))
    fig.savefig(plot_dir / f'scatter_compare-sub1.{ext}', dpi=300,
                bbox_inches=Bbox([[width/2,0],[width,height]]))
