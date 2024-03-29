#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-10-30-correlation-table.py: writes tables of the correlations for multi-
fidelity functions to small latex files and creates plot of correlation vs
tuning parameter.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import sys
from collections import namedtuple

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import mf2
import numpy as np
import pandas as pd
from parse import Parser
from pyprojroot import here
from scipy.stats import pearsonr, spearmanr

import processing as proc

sys.path.append(str(here()))

import multiLevelCoSurrogates as mlcs

np.random.seed(20160501)
np.set_printoptions(linewidth=200, edgeitems=5)

print(f'Running script: {__file__}')

save_dir = here() / "files/2019-10-30-correlations/"
save_dir.mkdir(parents=True, exist_ok=True)
table_dir = here() / "tables/2020-03-04-correlations/"
table_dir.mkdir(parents=True, exist_ok=True)
plot_dir = here() / "plots/2019-10-correlation-exploration/"
plot_dir.mkdir(parents=True, exist_ok=True)


def standardize_name(name):
    return name.lower().replace(' ', '-')


np.random.seed(20160501)  # standardize random seed for reproducibility

size_per_dim = 2000
dims = [1, 2, 3, 4, 6, 8, 10]
test_sample = {
    ndim: np.random.random((ndim * size_per_dim, ndim))
    for ndim in dims
}

Corr_result = namedtuple("Corr_result", "fname ndim pearson_r pearson_r2 spearman_r spearman_r2")
results = []
for i, f in enumerate(mf2.bi_fidelity_functions):
    sample = mlcs.rescale(test_sample[f.ndim], range_in=(0, 1), range_out=f.bounds)

    y_h, y_l = f.high(sample), f.low(sample)
    pear, spear = pearsonr(y_h, y_l)[0], spearmanr(y_h, y_l)[0]
    name = standardize_name(f.name)
    results.append(
        Corr_result(name, f.ndim, pear, pear**2, spear, spear**2)
    )

bi_fid_correlations = pd.DataFrame.from_records(results, columns=Corr_result._fields)
bi_fid_correlations = bi_fid_correlations.sort_values(by=['ndim', 'fname'])
bi_fid_correlations.to_latex(table_dir / 'correlations-table.tex',
                             float_format="{:0.3f}".format, index=False)


results = []
for ndim, sample in test_sample.items():
    sample = mlcs.rescale(sample, range_in=(0, 1), range_out=mf2.forrester.bounds)

    y_h, y_l = mf2.forrester.high(sample), mf2.forrester.low(sample)
    pear, spear = pearsonr(y_h, y_l)[0], spearmanr(y_h, y_l)[0]
    name = standardize_name(mf2.forrester.name)
    results.append(
        Corr_result(name, ndim, pear, pear**2, spear, spear**2)
    )

forrester_correlations = pd.DataFrame.from_records(results,
                                                   columns=Corr_result._fields)
forrester_correlations.to_latex(table_dir / 'forrester-correlations-table.tex',
                                float_format="{:0.3f}".format, index=False)


regular_correlations = pd.concat([bi_fid_correlations, forrester_correlations],
                                 ignore_index=True)
regular_correlations = regular_correlations.drop_duplicates()


params = np.round(np.linspace(0, 1, 101), 3)
Adj_Corr_result = namedtuple("Corr_result",
                             "fname ndim param pearson_r pearson_r2 spearman_r spearman_r2")

name_parser = Parser('Adjustable {fname} {param:f}')
results = []
for func in mf2.adjustable.bi_fidelity_functions:
    for a in params:
        f = func(np.round(a, 3))
        bounds = (np.array(f.l_bound), np.array(f.u_bound))
        sample = mlcs.rescale(test_sample[f.ndim], range_in=(0, 1), range_out=bounds)

        y_h, y_l = f.high(sample), f.low(sample)
        pear, spear = pearsonr(y_h, y_l)[0], spearmanr(y_h, y_l)[0]

        name = name_parser.parse(func(0.5).name)['fname']
        name = 'Hartmann3' if name == 'Hartmann' else name
        name = standardize_name(name)
        results.append(
            Adj_Corr_result(name, f.ndim, a, pear, pear**2, spear, spear**2)
        )

adjustables_correlations = pd.DataFrame.from_records(results, columns=Adj_Corr_result._fields)
adjustables_correlations.to_csv(save_dir / 'adjustables_correlations.csv', index=False)
adjustables_correlations.to_latex(table_dir / 'adjustables-correlations.tex',
                                  float_format="{:0.3f}".format, index=False)


all_correlations = pd.concat([regular_correlations, adjustables_correlations],
                             keys=['regular', 'adjustable'],
                             names=['category'],
                             sort=False,).reset_index('category')
all_correlations.to_csv(save_dir / 'correlations.csv', index=False)


styles = plt.rcParams['axes.prop_cycle'][:4] + cycler(linestyle=['-', '--', ':', '-.'])
plt.rc('axes', prop_cycle=styles)
figsize=(6.5, 3.3)
single_figsize = (3.25, 2.5)
labels = {
    'pearson_r': '$r$',
    'pearson_r2': '$r^2$',
    'spearman_r': 'Spearman $r$',
    'spearman_r2': 'Spearman $r^2$',
}

grouped_df = adjustables_correlations.groupby('fname')
fig = plt.figure(figsize=figsize)
gs = fig.add_gridspec(nrows=2, ncols=2, bottom=0.18, wspace=0.10, hspace=0.23, right=0.975, top=0.94, left=0.1)
axes = [
    fig.add_subplot(gs[0,0]),
    fig.add_subplot(gs[1,0]),
    fig.add_subplot(gs[0,1]),
    fig.add_subplot(gs[1,1]),
]


for i, (ax, (name, subdf)) in enumerate(zip(axes, grouped_df)):
    for col in ['pearson_r', 'pearson_r2']:  #, 'spearman_r', 'spearman_r2']:
        ax.plot(subdf['param'], subdf[col], label=labels[col])

    ax.axhline(y=0, color='black', alpha=.5)
    ax.set_xlim([0,1])
    ax.set_ylim([-1,1])
    ax.grid(alpha=.5, linestyle=':', which='both')
    if i < 2:
        ax.set_ylabel('Correlation')
        ax.yaxis.set_major_locator(MultipleLocator(.5))
    else:
        ax.yaxis.set_tick_params(left=False, labelleft=False, which='both')

    if i % 2 == 1:
        ax.set_xlabel('A')
        ax.xaxis.set_major_locator(MultipleLocator(.2))
    else:
        ax.xaxis.set_tick_params(left=False, labelleft=False, which='both')
    ax.set_title(name.title())

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.538, 0.0), ncol=len(handles))
for suffix in proc.suffixes:
    fig.savefig(plot_dir / f'combined_correlations{suffix}', dpi=300)

