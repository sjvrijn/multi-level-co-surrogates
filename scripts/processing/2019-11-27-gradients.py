#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-11-27-gradients.py: script to calculate some form of gradients for all
mse-grid files
"""

from collections import namedtuple
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
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
        with xr.open_dataset(file) as ds:
            da = ds['mses'].sel(model='high_hier').load()
        reg = proc.fit_lin_reg(da)
        coef = reg.coef_

        records.append(Record(directory.name, file.stem, *coef))

        # create lin-reg based plots
        #TODO: factor out into separate function
        n_high_y = np.array(da.coords['n_high'].values).reshape((-1, 1))
        n_low_x = np.array(da.coords['n_low'].values).reshape((1, -1))

        out = n_high_y*coef[0] + n_low_x*coef[1] + reg.intercept_
        out = np.triu(out)
        out[out == 0] = np.nan
        plt.imshow(out, origin='lower', cmap='viridis_r')
        plt.contour(out, levels=10, colors='black', alpha=.2, linewidths=1)
        plt.title(f"{directory.name}: {file.stem}")
        plt.show()


        da.close()



df = pd.DataFrame.from_records(records, columns=Record._fields)
print(df.to_latex())
