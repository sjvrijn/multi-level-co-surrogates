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

correlations = pd.read_csv(here('files') / 'correlations.csv')
extended_correlations_path = here('files') / 'extended_correlations.csv'




fname_template = re.compile(r'[A-Za-z]*-(\d+)d-Adjustable([A-Za-z]*3?)([01].\d+).nc')
Record = namedtuple('Record', 'orig_deg sub_deg cv_deg')
records = []
for file in sorted(adjustables_dir.iterdir()):
    match = fname_template.match(file.name)
    if not match:
        continue

    ndim, func_name, value = match.groups()

    orig_deg = correlations.loc[(correlations['name'] == func_name.lower())
                                & (correlations['param'] == float(value))]['deg']

    with xr.open_dataset(file) as ds:
        da = ds['mses'].sel(model='high_hier')
        with da.load() as da:
            reg = proc.fit_lin_reg(da)
        sub_deg = np.rad2deg(np.arctan2(*reg.coef_[:2])) % 180

        da = ds['cv_mses'].sel(model='high_hier')
        with da.load() as da:
            reg = proc.fit_lin_reg(da)
        cv_deg = np.rad2deg(np.arctan2(*reg.coef_[:2])) % 180

    records.append(Record(orig_deg, sub_deg, cv_deg))

df = pd.DataFrame(records)


fig, axes = plt.subplots(ncols=2, figsize=(9, 4.5), constrained_layout=True)

axes[0].scatter(df['orig_deg'], df['sub_deg'])
axes[0].plot([0, 100], '--', color='black', alpha=.3, linewidth=.5)
axes[0].set_title('Actual gradient vs sub-sampled gradient')


axes[1].scatter(df['orig_deg'], df['cv_deg'])
axes[1].plot([0, 100], '--', color='black', alpha=.3, linewidth=.5)
axes[1].set_title('Actual gradient vs cross-validated gradient')

plt.savefig(plot_dir / 'scatter_compare.pdf')
plt.savefig(plot_dir / 'scatter_compare.png')
