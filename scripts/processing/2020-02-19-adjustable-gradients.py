#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-02-19-adjustable-gradients.py: script to create scatterplots of the
gradients in the error versus the actual correlation between high- and low-
fidelity functions. Creates a csv of the all relevant values as intermediate
step to speed up any potential rerun.
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from pyprojroot import here

import processing as proc


__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

regulars_dir = here('files/2019-09-mse-nc/')
adjustables_dir = here("files/2019-10-07-adjustables/")
plot_dir = here('plots/2020-02-19-adjustable-gradients/', warn=False)
plot_dir.mkdir(exist_ok=True, parents=True)

correlations_path = here('files/correlations.csv')
extended_correlations_path = here('files/extended_correlations.csv', warn=False)

line_at_90 = dict(y=90, alpha=.5, color='black')
grid_style = dict(alpha=.5, linestyle=':')
xlabel = '{corr_type} Correlation $r$'
ylabel = 'Angle of Gradient'
correlation_range = np.linspace(-1, 1, 11)


def create_plots(correlations, angles, plot_individuals=False):
    """plot the error-grid gradient against correlation for the various
    (adjustable) benchmark functions. Creates plots for both pearson and
    spearman correlation, and for various groups of functions as well as a
    single overview plot. Shows estimated upper/lower range of angles based on
    CI of linear regression fit parameters as error bars."""

    correlation_types = ['pearson', 'spearman']

    extended_correlations = angles.merge(correlations)

    markers = 'ov^<>+x*1234'
    for corr_type in correlation_types:
        if plot_individuals:
            scatter_per_function(corr_type, extended_correlations)

        plt.subplots(figsize=(7.0,5.2), constrained_layout=True)

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
        plt.xticks(correlation_range)
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


def scatter_per_function(corr_type, extended_correlations):
    for category, df in extended_correlations.groupby('category'):
        for func_name, sub_df in df.groupby('fname'):
            x, y = sub_df[f'{corr_type}_r'].values, sub_df['deg'].values
            errors = np.stack([y - sub_df['deg_low'].values, sub_df['deg_high'].values - y])
            plt.errorbar(x, y, yerr=errors, ls='', capsize=1)
            plt.axhline(**line_at_90)
            plt.title(func_name)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(correlation_range)
            plt.grid(**grid_style)
            plt.tight_layout()
            plt.savefig(plot_dir / f'{category}-{func_name}-{corr_type}.png')
            plt.savefig(plot_dir / f'{category}-{func_name}-{corr_type}.pdf')
            plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--regen-csv', action='store_true')
    parser.add_argument('--no-plots', action='store_false')

    args = parser.parse_args()

    if args.regen_csv or not (adjustables_dir / 'gradients.csv').exists():
        proc.calc_and_store_gradient_angles(adjustables_dir)
    if args.regen_csv or not (regulars_dir / 'gradients.csv').exists():
        proc.calc_and_store_gradient_angles(regulars_dir)

    index_columns = ['category', 'ndim', 'fname', 'param']

    correlations = pd.read_csv(correlations_path, index_col=index_columns)
    adjustable_angles = pd.read_csv(adjustables_dir / 'gradients.csv', index_col=index_columns)
    regular_angles = pd.read_csv(regulars_dir / 'gradients.csv', index_col=index_columns)

    angles = regular_angles.append(adjustable_angles)

    if not args.no_plots:
        create_plots(correlations, angles)
