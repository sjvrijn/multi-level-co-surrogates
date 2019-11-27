#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-11-27-gradients.py: script to calculate some form of gradients for all
mse-grid files
"""

from collections import namedtuple
from itertools import product

import pandas as pd
import xarray as xr
from pyprojroot import here

import processing as proc


__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

data_dir = here("files")

Record = namedtuple('Record', 'experiment file high low rep')

records = []
for directory in [d for d in data_dir.iterdir() if d.is_dir()]:
    print(directory.name)

    for file in [f for f in directory.iterdir() if f.suffix == '.nc']:
        # TODO: properly use with-statement here?
        ds = xr.open_dataset(file)
        da = ds['mses'].load().sel(model='high_hier')
        coef = proc.fit_lin_reg_coefficients(da)
        da.close()

        records.append(Record(directory.name, file.name, *coef))

df = pd.DataFrame.from_records(records, columns=Record._fields)
print(df.to_latex())
