#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-02-19-adjustable0gradients.py: script to create scatterplots of the
gradients in the error versus the actual correlation between high- and low-
fidelity functions.
"""

from collections import defaultdict, namedtuple
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

fname_template = re.compile('[A-Za-z]*-(\d+)d-Adjustable([A-Za-z]*3?)([01].\d+).nc')
Record = namedtuple('Record', 'experiment file high low rep')
records = []

correlations = pd.read_csv(here('files') / 'correlations.csv', index_col=0)

to_plot = defaultdict(list)

for file in [f for f in data_dir.iterdir() if f.suffix == '.nc']:
    ndim, func_name, value = fname_template.match(file.name).groups()
    corrs = correlations.loc[(correlations['name'] == func_name.lower()) & (correlations['param'] == float(value)), 'pearson_r':'spearman_r2'].squeeze()

#     with xr.open_dataset(file) as ds:
#         da = ds['mses'].sel(model='high_hier')
#     with da.load() as da:
#         reg = proc.fit_lin_reg(da)
#     coef = reg.coef_
#     angle = ...

    angle = np.random.random()*90

    to_plot[func_name].append((corrs['pearson_r'], angle))


from pprint import pprint
pprint(to_plot)


for func_name, data in to_plot.items():
    x, y = zip(*data)
    plt.scatter(x, y)
    plt.title(func_name)
    plt.xlabel('correlation')
    plt.ylabel('angle')
    plt.tight_layout()
    plt.savefig(here('plots') / f'tmp/{func_name}.png')
    plt.close()