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
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import xarray as xr
from pyprojroot import here

import processing as proc


__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

regulars_dir = here('files/2019-09-mse-nc/')
adjustables_dir = here("files/2019-10-07-adjustables/")
plot_dir = here('plots/2020-02-19-adjustable-gradients/', warn=False)
plot_dir.mkdir(exist_ok=True, parents=True)

fname_template = re.compile(r'[A-Za-z]*-(\d+)d-Adjustable([A-Za-z]*3?)([01].\d+).nc')

correlations_path = here('files/correlations.csv')
extended_correlations_path = here('files/extended_correlations.csv', warn=False)


def calc_and_print_CIs(da):
    reg, SSE = proc.fit_lin_reg(da, calc_SSE=True)

    alpha, beta, gamma, epsilon = reg.coef_
    df = 4
    s_nhigh = np.std(da.coords['nhigh'], ddof=0)
    s_nlow = np.std(da.coords['nlow'], ddof=0)

    se_nhigh = np.sqrt(SSE / (da.size-df)) / s_nhigh
    se_nlow = np.sqrt(SSE / (da.size-df)) / s_nlow

    ci_alpha = [alpha-1.96*se_nhigh, alpha+1.96*se_nhigh]
    ci_beta = [beta-1.96*se_nlow, beta+1.96*se_nlow]
    ci_angle = [np.atan(ci_alpha[0]/ci_beta[1]),
                np.atan(ci_alpha[1]/ci_beta[0])]

    print(f'95% CI alpha: {ci_alpha}')
    print(f'95% CI beta: {ci_beta}')
    print(f'95% CI angle: {ci_angle}')



def store_extended_correlations():

    Record = namedtuple(

        'Record', 'category fname ndim pearson_r pearson_r2 '
                  'spearman_r spearman_r2 param '
                  'alpha beta gamma theta deg'
    )
    records = []
    correlations = pd.read_csv(correlations_path)

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

            calc_and_print_CIs(da)

    df = pd.DataFrame.from_records(records, columns=Record._fields)
    return df


if not extended_correlations_path.exists():
    extended_correlations = store_extended_correlations()
    extended_correlations.to_csv(extended_correlations_path, index=False)
else:
    extended_correlations = pd.read_csv(extended_correlations_path)


line_at_90 = dict(y=90, alpha=.5, color='black')
grid_style = dict(alpha=.5, linestyle=':')
xlabel = '{corr_type} Correlation $r$'
ylabel = 'Angle of Gradient'

xticks = np.linspace(-1, 1, 11)
correlation_types = ['pearson', 'spearman']


for corr_type in correlation_types:
    for category, df in extended_correlations.groupby('category'):
        for func_name, sub_df in df.groupby('fname'):
            x, y = sub_df[f'{corr_type}_r'].values, sub_df['deg'].values
            plt.scatter(x, y)
            plt.axhline(**line_at_90)
            plt.title(func_name)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(xticks)
            plt.grid(**grid_style)
            plt.tight_layout()
            plt.savefig(plot_dir / f'{category}-{func_name}.png')
            plt.savefig(plot_dir / f'{category}-{func_name}.pdf')
            plt.close()


    fig, ax = plt.subplots(figsize=(7.0,5.2), constrained_layout=True)

    adjustables = extended_correlations.loc[extended_correlations['category'] == 'adjustable']
    for func_name, sub_df in adjustables.groupby('fname'):
        x, y = sub_df[f'{corr_type}_r'].values, sub_df['deg'].values
        #plt.scatter(x, y, marker='.', label=func_name)
        plt.plot(x, y, linestyle='-', linewidth=.3, marker='.', label=f'adjustable {func_name}')


    regulars = extended_correlations.loc[extended_correlations['category'] == 'regular']
    markers = 'ov^<>+x*1234'
    for (func_name, sub_df), marker in zip(regulars.groupby('fname'), markers):
        x, y = sub_df[f'{corr_type}_r'].values, sub_df['deg'].values
        plt.scatter(x, y, marker=marker, label=func_name)


    plt.axhline(**line_at_90)
    plt.title("Comparing Adjustable Functions")
    plt.xlabel(xlabel.format(corr_type=corr_type.title()))
    plt.ylabel(ylabel)
    plt.xticks(xticks)
    ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    plt.xlim([-1, 1.05])
    plt.ylim([0, 120])
    plt.grid(**grid_style, which='both')

    plt.legend(loc='lower left')
    # Shrink current axis by 20%
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    # Put a legend to the right of the current axis
    #ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(plot_dir / f'comparison_{corr_type}.png')
    plt.savefig(plot_dir / f'comparison_{corr_type}.pdf')
    plt.close()
