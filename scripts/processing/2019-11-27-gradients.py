#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2019-11-27-gradients.py: script to calculate some form of gradients for all
mse-grid files
#TODO: review necessity of this file
"""

from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pyprojroot import here

import processing as proc

print(f'Running script: {__file__}')

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

data_dir = here("files")
save_dir = here() / "tables/2019-11-27-gradients/"
save_dir.mkdir(parents=True, exist_ok=True)


Record = namedtuple('Record', 'experiment file high low')

records = []
for directory in [d for d in data_dir.iterdir() if d.is_dir()]:
    print(directory.name)

    for file in [f for f in directory.iterdir() if f.suffix == '.nc']:
        try:
            with xr.open_dataset(file) as ds:
                da = ds['mses'].sel(model='high_hier')
        except KeyError:
            continue

        with da.load() as da:
            reg = proc.fit_lin_reg(da)
            coef = reg.coef_
            records.append(Record(directory.name, file.stem, *coef))
            n_high_y = np.array(da.coords['n_high'].values).reshape((-1, 1))
            n_low_x = np.array(da.coords['n_low'].values).reshape((1, -1))

        # create lin-reg based plots
        out = n_high_y*coef[0] + n_low_x*coef[1] + reg.intercept_
        out = np.triu(out)
        out[out == 0] = np.nan
        #TODO: make plot optional
        plt.imshow(out, origin='lower', cmap='viridis_r')
        plt.contour(out, levels=10, colors='black', alpha=.2, linewidths=1)
        plt.title(f"{directory.name}: {file.stem}")
        plt.show()

#TODO: store as csv
df = pd.DataFrame.from_records(records, columns=Record._fields)
df.to_latex(save_dir / 'all_gradients.tex',
            float_format="{:0.3f}".format, index=False)
