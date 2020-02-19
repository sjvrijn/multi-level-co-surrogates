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

fname_template = re.compile('[A-Za-z]*-(\d+)d-Adjustable([A-Za-z]*3?)([01].\d+).nc')
Record = namedtuple('Record', 'experiment file high low rep')
records = []

for file in [f for f in data_dir.iterdir() if f.suffix == '.nc']:
    ndim, func_name, value = fname_template.match(file.name).groups()

    # with xr.open_dataset(file) as ds:
    #     da = ds['mses'].sel(model='high_hier')
    # with da.load() as da:
    #     reg = proc.fit_lin_reg(da)
    #     coef = reg.coef_



        # records.append(Record(directory.name, file.stem, *coef))


df = pd.DataFrame.from_records(records, columns=Record._fields)
print(df.to_latex())
