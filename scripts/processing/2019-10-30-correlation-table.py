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


size_per_dim = 2000
dims = [1, 2, 3, 4, 6, 8, 10]
test_sample = {
    ndim: np.random.random((ndim * size_per_dim, ndim))
    for ndim in dims
}

Corr_result = namedtuple("Corr_result", "name ndim pearson_r pearson_r2 spearman_r spearman_r2")
results = []
for i, f in enumerate(mf2.bi_fidelity_functions):
    if i == 5:  # TODO: remove mf2.Forrester from bi_fidelity_functions
        continue
    sample = mlcs.rescale(test_sample[f.ndim], range_in=(0, 1), range_out=f.bounds)

    y_h, y_l = f.high(sample), f.low(sample)
    pear, spear = pearsonr(y_h, y_l)[0], spearmanr(y_h, y_l)[0]
    results.append(Corr_result(f.name.lower(), f.ndim, pear, pear**2, spear, spear**2))

bi_fid_correlations = pd.DataFrame.from_records(results, columns=Corr_result._fields)
bi_fid_correlations = bi_fid_correlations.sort_values(by=['ndim', 'name'])
bi_fid_correlations.to_latex(save_dir / 'correlations-table.tex', index=False)


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
                                index=False)


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
adjustables_correlations.to_latex(save_dir / 'adjustables-correlations.tex')


all_correlations = pd.concat([regular_correlations, adjustables_correlations],
                             keys=['regular', 'adjustable'],
                             names=['category'],
                             sort=False,).reset_index('category')
all_correlations.to_csv(here('files') / 'correlations.csv', index=False)
