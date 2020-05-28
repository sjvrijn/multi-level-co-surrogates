#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-10-30-correlation-table.py: writes tables of the correlations for multi-
fidelity functions to small latex files.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import sys
from collections import namedtuple

from cycler import cycler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyprojroot import here
from scipy.stats import pearsonr, spearmanr

sys.path.append(str(here()))

import mf2
import multiLevelCoSurrogates as mlcs

np.set_printoptions(linewidth=200, edgeitems=5)

save_dir = here() / "tables/2020-03-04-correlations/"
save_dir.mkdir(parents=True, exist_ok=True)
plot_dir = here() / "plots/2019-10-correlation-exploration/"
plot_dir.mkdir(parents=True, exist_ok=True)


size_per_dim = 2000
dims = [1, 2, 3, 4, 6, 8, 10]
test_sample = {
    ndim: np.random.random((ndim * size_per_dim, ndim))
    for ndim in dims
}

Corr_result = namedtuple("Corr_result", "name ndim pearson_r pearson_r2 spearman_r spearman_r2")
results = []
for i, f in enumerate(mf2.bi_fidelity_functions):
    sample = mlcs.rescale(test_sample[f.ndim], range_in=(0, 1), range_out=f.bounds)

    y_h, y_l = f.high(sample), f.low(sample)
    pear, spear = pearsonr(y_h, y_l)[0], spearmanr(y_h, y_l)[0]
    results.append(Corr_result(f.name.lower(), f.ndim, pear, pear**2, spear, spear**2))

bi_fid_correlations = pd.DataFrame.from_records(results, columns=Corr_result._fields)
bi_fid_correlations = bi_fid_correlations.sort_values(by=['ndim', 'name'])
bi_fid_correlations.to_latex(save_dir / 'correlations-table.tex',
                             float_format="{:0.3f}".format, index=False)


results = []
for ndim, sample in test_sample.items():
    sample = mlcs.rescale(sample, range_in=(0, 1), range_out=mf2.forrester.bounds)

    y_h, y_l = mf2.forrester.high(sample), mf2.forrester.low(sample)
    pear, spear = pearsonr(y_h, y_l)[0], spearmanr(y_h, y_l)[0]
    results.append(
        Corr_result(mf2.forrester.name.lower(), ndim, pear, pear**2, spear, spear**2))

forrester_correlations = pd.DataFrame.from_records(results,
                                                   columns=Corr_result._fields)
forrester_correlations.to_latex(save_dir / 'forrester-correlations-table.tex',
                                float_format="{:0.3f}".format, index=False)


regular_correlations = pd.concat([bi_fid_correlations, forrester_correlations],
                                 ignore_index=True)
regular_correlations = regular_correlations.drop_duplicates()


params = np.round(np.linspace(0, 1, 101), 3)
Adj_Corr_result = namedtuple("Corr_result",
                             "name ndim param pearson_r pearson_r2 spearman_r spearman_r2")

results = []
for func in mf2.adjustable.bifidelity_functions:
    for a in params:
        f = func(np.round(a, 3))
        bounds = (np.array(f.l_bound), np.array(f.u_bound))
        sample = mlcs.rescale(test_sample[f.ndim], range_in=(0, 1), range_out=bounds)

        y_h, y_l = f.high(sample), f.low(sample)
        pear, spear = pearsonr(y_h, y_l)[0], spearmanr(y_h, y_l)[0]
        results.append(
            Adj_Corr_result(func.__name__, f.ndim, a, pear, pear**2, spear, spear**2))

adjustables_correlations = pd.DataFrame.from_records(results, columns=Adj_Corr_result._fields)
adjustables_correlations.to_csv(here('files') / 'adjustables_correlations.csv', index=False)
adjustables_correlations.to_latex(save_dir / 'adjustables-correlations.tex',
                                  float_format="{:0.3f}".format, index=False)


all_correlations = pd.concat([regular_correlations, adjustables_correlations],
                             keys=['regular', 'adjustable'],
                             names=['category'],
                             sort=False,).reset_index('category')
all_correlations.to_csv(here('files') / 'correlations.csv', index=False)


styles = plt.rcParams['axes.prop_cycle'][:4] + cycler(linestyle=['-', '--', ':', '-.'])
plt.rc('axes', prop_cycle=styles)
figsize=(7.2, 5.6)
single_figsize = (3.6, 2.7)
labels = {
    'pearson_r': 'Pearson $r$',
    'pearson_r2': 'Pearson $r^2$',
    'spearman_r': 'Spearman $r$',
    'spearman_r2': 'Spearman $r^2$',
}

param_idx = {'branin': 1, 'paciorek': 2, 'hartmann3': 3, 'trid': 4}

grouped_df = adjustables_correlations.groupby('name')
fig = plt.figure(figsize=figsize)  #, constrained_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=2, bottom=0.16, wspace=0.3, hspace=0.45, right=0.975, top=0.95, left=0.1)
axes = [
    fig.add_subplot(gs[0,0]),
    fig.add_subplot(gs[1,0]),
    fig.add_subplot(gs[0,1]),
    fig.add_subplot(gs[1,1]),
]

for ax_i, (name, subdf) in zip(axes, grouped_df):
    single_fig, single_ax = plt.subplots(figsize=single_figsize, constrained_layout=True)

    for ax in [single_ax, ax_i]:
        for col in ['pearson_r', 'pearson_r2', 'spearman_r', 'spearman_r2']:
            ax.plot(subdf['param'], subdf[col], label=labels[col])

        ax.axhline(y=0, color='black', alpha=.5)
        ax.set_xlim([0,1])
        ax.set_ylim([-1,1])
        ax.set_ylabel('Correlation')
        ax.set_xlabel(f'A{param_idx[name]}')
        ax.set_title(name.title())

    single_ax.legend(loc=0)
    single_fig.savefig(plot_dir / f'{name}_correlation.pdf')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=len(handles))
#fig.tight_layout()
fig.savefig(plot_dir / 'combined_correlations.pdf')

