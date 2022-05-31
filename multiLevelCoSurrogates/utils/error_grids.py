from collections import namedtuple

import numpy as np
from sklearn.linear_model import LinearRegression
import xarray as xr

import multiLevelCoSurrogates as mlcs


def fit_lin_reg(da: xr.DataArray, calc_SSE: bool=False):
    """Determine linear regression coefficients after training index -> value"""

    series = da.to_series().dropna()
    X = np.array(series.index.tolist())[:,:2]  # remove rep_idx (3rd column)
    y = np.log10(series.values)
    reg = LinearRegression().fit(X, y)

    if not calc_SSE:
        return reg

    pred_y = reg.predict(X)
    SSE = np.sum((pred_y - y)**2)
    return reg, SSE


def calc_angle(da: xr.DataArray):
    """Calculate the global gradient angle of an Error Grid based
    on the slope of beta_1 / beta_2 from a linear regression fit.
    """
    AngleSummary = namedtuple('AngleSummary', 'alpha beta theta deg deg_low deg_high')
    reg, SSE = mlcs.utils.error_grids.fit_lin_reg(da, calc_SSE=True)

    beta_high, beta_low = reg.coef_
    ratio = beta_high / beta_low
    df = da.size - 3

    nhighs = da.coords['n_high'].values
    var_nhigh = np.sqrt(np.sum((nhighs - np.mean(nhighs))**2))

    nlows = da.coords['n_low'].values
    var_nlow = np.sqrt(np.sum((nlows - np.mean(nlows))**2))

    s = np.sqrt(SSE / df)

    se_nhigh = s / var_nhigh
    se_nlow = s / var_nlow
    se_ratio = np.sqrt((se_nhigh / beta_high)**2 + (se_nlow / beta_low)**2)

    ci_beta_high = ConfidenceInterval(beta_high, se_nhigh, beta_high-1.96*se_nhigh, beta_high+1.96*se_nhigh)
    ci_beta_low = ConfidenceInterval(beta_low, se_nlow, beta_low-1.96*se_nlow, beta_low+1.96*se_nlow)
    ci_beta_ratio = ConfidenceInterval(ratio, se_ratio, ratio-1.96*se_ratio, ratio+1.96*se_ratio)

    theta = np.arctan(ratio)
    mid_angle = np.rad2deg(theta)

    min_angle = np.rad2deg(np.arctan(ci_beta_ratio.lower))
    max_angle = np.rad2deg(np.arctan(ci_beta_ratio.upper))
    if mid_angle < 0:
        min_angle, mid_angle, max_angle = min_angle+180, mid_angle+180, max_angle+180
    elif mid_angle > 180:
        min_angle, mid_angle, max_angle = min_angle-180, mid_angle-180, max_angle-180

    return AngleSummary(beta_high, beta_low, theta, mid_angle, min_angle, max_angle)


class ConfidenceInterval(namedtuple('ConfidenceInterval', 'mean se lower upper')):

    def __contains__(self, value: float):
        return self.lower < value < self.upper

    def __str__(self):
        lower = self.lower if self.lower is not None else self.mean - 1.96*self.se
        upper = self.upper if self.upper is not None else self.mean + 1.96*self.se
        return f'95% CI: {self.mean:.4f} +/- {1.96*self.se:.4f} {np.array([lower, upper])}: ' \
               f'H0{" not" if 0 in self else ""} rejected'