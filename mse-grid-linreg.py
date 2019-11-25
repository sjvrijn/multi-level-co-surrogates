# coding: utf-8
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.linear_model import LinearRegression


def run(da):
    """Return lin-reg coefficients after training index -> value"""

    # clean data
    series = da.to_series().dropna()
    # extract X and y
    X = np.array(series.index.tolist())
    y = series.values
    # compute and return coefficients
    return LinearRegression().fit(X, y).coef_

if __name__ == '__main__':
    with xr.open_dataset("Matern-1d-Forrester-sub50-125.nc") as ds:
        da = ds['mses'].load().sel(model='high_hier')
    run(da)


