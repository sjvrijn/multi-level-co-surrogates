#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-03-09-subsampling-gradients.py: script to create scatterplots of the
gradients in the error for the actual, subsampled, and cross-validated MSE
measures.
Creates a csv of the all relevant values as intermediate step to speed up any
potential rerun.
"""

from collections import namedtuple
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyprojroot import here

import processing as proc


__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

adjustables_dir = here("files/2019-10-07-adjustables/")
subsampling_dir = here("files/2020-03-04-cv-adjustables-subsampling")
plot_dir = here("plots") / '2020-03-09-subsampling-gradients/'
plot_dir.mkdir(exist_ok=True, parents=True)

correlations = pd.read_csv(here('files') / 'extended_correlations.csv')



fname_template = re.compile(r'[A-Za-z]*-(\d+)d-Adjustable([A-Za-z]*3?)([01].\d+)-sub50-125-seed(\d).nc')
Record = namedtuple('Record', 'name param seed_offset pearson_r orig_deg sub_deg cv_deg')

save_extensions = ['pdf', 'png']

records = []
names = set()
for file in sorted(subsampling_dir.iterdir()):
    match = fname_template.match(file.name)
    if not match:
        continue

    ndim, func_name, param, seed_offset = match.groups()
    names.add(func_name)
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

df = pd.DataFrame(records)

grid_style = dict(b=True, alpha=.5, linestyle=':')
scatter_style = {'s': 12}
diag_line = {'linestyle': '--', 'color': 'black', 'alpha':.3, 'linewidth':.5}
fig, axes = plt.subplots(ncols=2, figsize=(9, 4.5), constrained_layout=True)


for name, sub_df in df.groupby('name'):
    axes[0].scatter(sub_df['orig_deg'], sub_df['sub_deg'], label=name, **scatter_style)
    axes[1].scatter(sub_df['orig_deg'], sub_df['cv_deg'], label=name, **scatter_style)

axes[0].set_title('Actual gradient vs sub-sampled gradient')
axes[0].set_ylabel('subsampled gradient')
axes[0].legend(loc=0)

axes[1].set_title('Actual gradient vs cross-validated gradient')
axes[1].set_ylabel('cross-validated gradient')

for idx, ax in enumerate(axes):
    ax.set_xlabel('actual gradient')
    ax.grid(**grid_style)
    ax.plot(np.arange(100), **diag_line)
    ax.set_xlim([0,90])
    ax.set_ylim([0,90])

    sep_fig = plt.figure()
    sep_fig.axes.append(ax)
    for extension in save_extensions:
        sep_fig.tight_layout()
        sep_fig.savefig(plot_dir / f'scatter_compare-sub{idx}.{extension}')

for extension in save_extensions:
    fig.savefig(plot_dir / f'scatter_compare.{extension}')
