#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-02-19-adjustable-gradients.py: script to create scatterplots of the
gradients in the error versus the actual correlation between high- and low-
fidelity functions. Creates a csv of the all relevant values as intermediate
step to speed up any potential rerun.
"""

from collections import namedtuple
from itertools import product
import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import xarray as xr
from pyprojroot import here
from parse import compile

import processing as proc


__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

regulars_dir = here('files/2019-09-mse-nc/')
adjustables_dir = here("files/2019-10-07-adjustables/")
plot_dir = here('plots/2020-02-19-adjustable-gradients/', warn=False)
plot_dir.mkdir(exist_ok=True, parents=True)

correlations_path = here('files/correlations.csv')
extended_correlations_path = here('files/extended_correlations.csv', warn=False)



class ConfidenceInterval(namedtuple('ConfidenceInterval', 'mean se lower upper')):

    def __contains__(self, value):
        return self.lower < value < self.upper

    def __str__(self):
        lower = self.lower if self.lower is not None else self.mean - 1.96*self.se
        upper = self.upper if self.upper is not None else self.mean + 1.96*self.se
        return f'95% CI: {self.mean:.4f} +/- {1.96*self.se:.4f} {np.array([lower, upper])}: H0{" not" if 0 in self else ""} rejected'


def ratio_to_angle(x1, x2):
    theta = np.arctan2(x1, x2) + np.pi
    deg = np.rad2deg(theta)
    if deg > 180:
        deg -= 360  # Example: instead of 350, we want -10
    return deg


def calc_angle(da):
    AngleSummary = namedtuple('AngleSummary', 'alpha beta theta deg deg_low deg_high')
    reg, SSE = proc.fit_lin_reg(da, calc_SSE=True)

    alpha, beta = reg.coef_
    df = da.size - 3

    nhighs = da.coords['n_high'].values
    var_nhigh = np.sqrt(np.sum((nhighs - np.mean(nhighs))**2))

    nlows = da.coords['n_low'].values
    var_nlow = np.sqrt(np.sum((nlows - np.mean(nlows))**2))

    s = np.sqrt(SSE / df)

    se_nhigh = s / var_nhigh
    se_nlow = s / var_nlow

    ci_alpha = ConfidenceInterval(alpha, se_nhigh, alpha-1.96*se_nhigh, alpha+1.96*se_nhigh)
    ci_beta = ConfidenceInterval(beta, se_nlow, beta-1.96*se_nlow, beta+1.96*se_nlow)

    theta = np.arctan2(alpha, beta) + np.pi
    mid_angle = np.rad2deg(theta)
    angles = [ratio_to_angle(a, b) for a, b in product(ci_alpha[2:], ci_beta[2:])]

    return AngleSummary(alpha, beta, theta, mid_angle, min(angles), max(angles))


def determine_extended_correlations():

    Record = namedtuple(

        'Record', 'category fname ndim pearson_r pearson_r2 '
                  'spearman_r spearman_r2 param '
                  'alpha beta theta deg deg_low deg_high'
    )
    records = []
    correlations = pd.read_csv(correlations_path)

    for category, directory in zip(('regular', 'adjustable'),
                                   [regulars_dir, adjustables_dir]):
        if category == 'regular':
            fname_template = compile("{surrogate:w}-{ndim:d}d-{fname}.nc")
        else:
            fname_template = compile("{surrogate:w}-{ndim:d}d-Adjustable{fname}{param:f}.nc")

        for file in sorted(directory.iterdir()):
            match = fname_template.parse(file.name)
            if not match:
                continue

            if category == 'regular':
                row = correlations.loc[(correlations['category'] == 'regular')
                                       & (correlations['name'] == match['fname'].lower())
                                       & (correlations['ndim'] == match['ndim'])].squeeze()
            else:
                row = correlations.loc[(correlations['name'] == match['fname'].lower())
                                       & (correlations['param'] == match['param'])].squeeze()

            with xr.open_dataset(file) as ds:
                da = ds['mses'].sel(model='high_hier')
            with da.load() as da:
                angle_summary = calc_angle(da)
            records.append(Record(*row, *angle_summary))

    return pd.DataFrame.from_records(records, columns=Record._fields)


def create_plots(extended_correlations):
    """plot the error-grid gradient against correlation for the various
    (adjustable) benchmark functions. Creates plots for both pearson and
    spearman correlation, and for various groups of functions as well as a
    single overview plot. Shows estimated upper/lower range of angles based on
    CI of linear regression fit parameters as error bars."""

    line_at_90 = dict(y=90, alpha=.5, color='black')
    grid_style = dict(alpha=.5, linestyle=':')
    xlabel = '{corr_type} Correlation $r$'
    ylabel = 'Angle of Gradient'

    xticks = np.linspace(-1, 1, 11)
    correlation_types = ['pearson', 'spearman']


    markers = 'ov^<>+x*1234'
    for corr_type in correlation_types:
        for category, df in extended_correlations.groupby('category'):
            for func_name, sub_df in df.groupby('fname'):
                x, y = sub_df[f'{corr_type}_r'].values, sub_df['deg'].values
                errors = np.stack([y - sub_df['deg_low'].values, sub_df['deg_high'].values - y])
                plt.errorbar(x, y, yerr=errors, ls='', capsize=1)
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
            errors = np.stack([y - sub_df['deg_low'].values, sub_df['deg_high'].values - y])
            plt.errorbar(x, y, yerr=errors, capsize=1, linestyle='-', linewidth=.5, marker='.', label=f'adjustable {func_name}')


        regulars = extended_correlations.loc[extended_correlations['category'] == 'regular']
        for (func_name, sub_df), marker in zip(regulars.groupby('fname'), markers):
            if func_name == 'forrester':
                sub_df = sub_df.loc[sub_df['ndim'] == 1]
            x, y = sub_df[f'{corr_type}_r'].values, sub_df['deg'].values
            errors = np.stack([y - sub_df['deg_low'].values, sub_df['deg_high'].values - y])
            plt.errorbar(x, y, yerr=errors, ls='', capsize=1, linewidth=.5, marker=marker, label=func_name)


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

        plt.savefig(plot_dir / f'comparison_{corr_type}.png')
        plt.savefig(plot_dir / f'comparison_{corr_type}.pdf')
        plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--regen-csv', action='store_true')
    parser.add_argument('--no-plots', action='store_false')

    args = parser.parse_args()

    if args.regen_csv or not extended_correlations_path.exists():
        ext_correlations = determine_extended_correlations()
        ext_correlations.to_csv(extended_correlations_path, index=False)
    else:
        ext_correlations = pd.read_csv(extended_correlations_path)

    if not args.no_plots:
        create_plots(ext_correlations)
