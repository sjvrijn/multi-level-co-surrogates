#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2020-09-04-linreg-testing-interaction.py: script to test if an interaction
parameter (nhigh * nlow) would add accuracy to a linear regression model.
"""

import numpy as np
import xarray as xr
from parse import Parser
from pyprojroot import here
from sklearn.linear_model import LinearRegression

print(f'Running script: {__file__}')


__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

regulars_dir = here('files/2019-09-mse-nc/')
adjustables_dir = here("files/2019-10-07-adjustables/")


def test_interaction(da: xr.DataArray):

    series = da.to_series().dropna()
    X = np.array(series.index.tolist())
    y = np.log10(series.values)
    reg = LinearRegression().fit(X[:,:2], y)
    pred_y = reg.predict(X[:,:2])
    SSE = np.sum((pred_y - y)**2)



    interaction = (X[:,0]*X[:,1]).reshape((-1,1))
    extended_X = np.hstack([X, interaction])
    ext_reg = LinearRegression().fit(extended_X, y)

    pred_ext_y = ext_reg.predict(extended_X)
    ext_SSE = np.sum((pred_ext_y - y)**2)

    print(f'{(1-(ext_SSE/SSE))*100:.2f}%')


    a, b, c, beta = ext_reg.coef_
    #print(f'{a/beta:.3f}, {b/beta:.3f}, {c/beta:.3f}')
    df = da.size - 5

    var_interaction = np.sqrt(np.sum((interaction - np.mean(interaction))**2))
    s = np.sqrt(ext_SSE / df)
    se = s / var_interaction
    return (beta - 1.96*se) < 0 < (beta + 1.96*se), beta, se



def check_for_interactions():

    for category, directory in zip(('regular', 'adjustable'),
                                   [regulars_dir, adjustables_dir]):
        if category == 'regular':
            fname_parser = Parser("{surrogate:w}-{ndim:d}d-{fname}.nc")
        else:
            fname_parser = Parser("{surrogate:w}-{ndim:d}d-Adjustable{fname}{param:f}.nc")

        for file in sorted(directory.iterdir()):
            match = fname_parser.parse(file.name)
            if not match:
                continue

            with xr.open_dataset(file) as ds:
                da = ds['mses'].sel(model='high_hier')
            with da.load() as da:
                print(f'{file.stem}:', end=' ')
                is_interaction_zero, b, se = test_interaction(da)

check_for_interactions()
