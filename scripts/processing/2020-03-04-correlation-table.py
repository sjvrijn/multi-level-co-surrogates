#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-03-04-correlation-table.py: writes tables of the correlations for multi-
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
    results.append(Corr_result(f.name, f.ndim, pear, pear**2, spear, spear**2))

df = pd.DataFrame.from_records(results, columns=Corr_result._fields)
df = df.sort_values(by=['ndim', 'name'])
df.to_latex(save_dir / 'correlations-table.tex', index=False)


results = []
for ndim, sample in test_sample.items():
    sample = mlcs.rescale(sample, range_in=(0, 1), range_out=mf2.forrester.bounds)

    y_h, y_l = mf2.forrester.high(sample), mf2.forrester.low(sample)
    pear, spear = pearsonr(y_h, y_l)[0], spearmanr(y_h, y_l)[0]
    results.append(
        Corr_result(mf2.forrester.name, ndim, pear, pear**2, spear, spear**2))

df = pd.DataFrame.from_records(results, columns=Corr_result._fields)
df.to_latex(save_dir / 'forrester-correlations-table.tex', index=False)
