#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-02-19-adjustable-gradients.py: script to create scatterplots of the
gradients in the error versus the actual correlation between high- and low-
fidelity functions. Creates a csv of the all relevant values as intermediate
step to speed up any potential rerun.
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

regulars_dir = here('files/2019-09-mse-nc/')
adjustables_dir = here("files/2019-10-07-adjustables/")
plot_dir = here("plots") / '2020-02-19-adjustable-gradients/'
plot_dir.mkdir(exist_ok=True, parents=True)

fname_template = re.compile(r'[A-Za-z]*-(\d+)d-Adjustable([A-Za-z]*3?)([01].\d+).nc')

correlations = pd.read_csv(here('files') / 'correlations.csv')
extended_correlations_path = here('files') / 'extended_correlations.csv'


def store_extended_correlations():

    Record = namedtuple(

        'Record', 'category fname ndim pearson_r pearson_r2 '
                  'spearman_r spearman_r2 param '
                  'alpha beta gamma theta deg'
    )
    records = []

    for category, directory in zip(('regular', 'adjustable'),
                                   [regulars_dir, adjustables_dir]):
        if category == 'regular':
            fname_template = re.compile(r'[A-Za-z]*-(\d+)d-([A-Za-z0-9]*).nc')
        else:
            fname_template = re.compile(r'[A-Za-z]*-(\d+)d-Adjustable([A-Za-z]*3?)([01].\d+).nc')

        for file in sorted(directory.iterdir()):
            match = fname_template.match(file.name)
            if not match:
                continue

            if category == 'regular':
                ndim, func_name = match.groups()
                row = correlations.loc[(correlations['category'] == 'regular')
                                       & (correlations['name'] == func_name.lower())
                                       & (correlations['ndim'] == int(ndim))].squeeze()
            else:
                ndim, func_name, value = match.groups()
                row = correlations.loc[(correlations['name'] == func_name.lower())
                                       & (correlations['param'] == float(value))].squeeze()

            with xr.open_dataset(file) as ds:
                da = ds['mses'].sel(model='high_hier')
            with da.load() as da:
                reg = proc.fit_lin_reg(da)
            coef = reg.coef_
            theta = np.arctan2(*reg.coef_[:2])
            deg = np.rad2deg(theta) % 180
            records.append(Record(*row, *coef, theta, deg))

    df = pd.DataFrame.from_records(records, columns=Record._fields)
    df.to_csv(here('files') / 'extended_correlations.csv', index=False)
    return df


if not extended_correlations_path.exists():
    extended_correlations = store_extended_correlations()
else:
    extended_correlations = pd.read_csv(extended_correlations_path)


line_at_90 = dict(y=90, alpha=.5, color='black')
grid_style = dict(b=True, alpha=.5, linestyle=':')
xlabel = 'Correlation $r$'
ylabel = 'Angle of Gradient'

xticks = np.linspace(-1, 1, 11)



for func_name, sub_df in extended_correlations.groupby('fname'):
    x, y = sub_df['pearson_r'].values, sub_df['deg'].values
    plt.scatter(x, y)
    plt.axhline(**line_at_90)
    plt.title(func_name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xticks)
    plt.grid(**grid_style)
    plt.tight_layout()
    plt.savefig(plot_dir / f'{func_name}.png')
    plt.savefig(plot_dir / f'{func_name}.pdf')
    plt.close()


for func_name, sub_df in extended_correlations.groupby('fname'):
    x, y = sub_df['pearson_r'].values, sub_df['deg'].values
    plt.scatter(x, y, label=func_name)


plt.axhline(**line_at_90)
plt.title("Comparing Adjustable Functions")
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.xticks(xticks)
plt.grid(**grid_style)
plt.legend(loc=0)
plt.tight_layout()
plt.savefig(plot_dir / 'comparison.png')
plt.savefig(plot_dir / 'comparison.pdf')
plt.close()
