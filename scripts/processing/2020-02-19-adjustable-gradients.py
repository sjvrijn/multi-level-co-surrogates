#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-02-19-adjustable0gradients.py: script to create scatterplots of the
gradients in the error versus the actual correlation between high- and low-
fidelity functions.
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

data_dir = here("files/2019-10-07-adjustables")
plot_dir = here("plots") / '2020-02-19-adjustable-gradients/'
plot_dir.mkdir(exist_ok=True, parents=True)

fname_template = re.compile(r'[A-Za-z]*-(\d+)d-Adjustable([A-Za-z]*3?)([01].\d+).nc')

correlations = pd.read_csv(here('files') / 'correlations.csv', index_col=0)
extended_correlations_path = here('files') / 'extended_correlations.csv'



def store_extended_correlations():

    Record = namedtuple('Record', 'fname ndim param pearson_r pearson_r2 spearman_r spearman_r2 alpha beta gamma theta')
    records = []

    for file in [f for f in sorted(data_dir.iterdir()) if f.suffix == '.nc']:
        match = fname_template.match(file.name)
        if not match:
            continue

        ndim, func_name, value = match.groups()
        row = correlations.loc[(correlations['name'] == func_name.lower()) & (correlations['param'] == float(value))].squeeze()

        with xr.open_dataset(file) as ds:
            da = ds['mses'].sel(model='high_hier')
        with da.load() as da:
            reg = proc.fit_lin_reg(da)
        coef = reg.coef_
        angle = np.arctan2(*reg.coef_[:2])
        records.append(Record(*row, *coef, angle))

    pd.DataFrame.from_records(records, columns=Record._fields).to_csv(here('files') / 'extended_correlations.csv', index=False)


if not extended_correlations_path.exists():
    store_extended_correlations()

extended_correlations = pd.read_csv(extended_correlations_path)


for func_name, sub_df in extended_correlations.groupby('fname'):
    x, y = sub_df['pearson_r'].values, sub_df['theta'].values
    y = np.rad2deg(y) % 180
    plt.scatter(x, y)
    plt.axhline(y=90, alpha=.5, color='black')
    plt.title(func_name)
    plt.xlabel('Correlation $r$')
    plt.ylabel('Angle of gradient')
    plt.tight_layout()
    plt.savefig(plot_dir / f'{func_name}.png')
    plt.savefig(plot_dir / f'{func_name}.pdf')
    plt.close()


for func_name, sub_df in extended_correlations.groupby('fname'):
    x, y = sub_df['pearson_r'].values, sub_df['theta'].values
    y = np.rad2deg(y) % 180
    plt.scatter(x, y, label=func_name)


plt.axhline(y=90, alpha=.5, color='black')
plt.title("Comparing Adjustable Functions")
plt.xlabel('Correlation $r$')
plt.ylabel('Angle of gradient')
plt.legend(loc=0)
plt.tight_layout()
plt.savefig(plot_dir / 'comparison.png')
plt.savefig(plot_dir / 'comparison.pdf')
plt.close()
