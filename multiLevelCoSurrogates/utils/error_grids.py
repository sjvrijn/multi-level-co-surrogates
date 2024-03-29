from _warnings import warn
from collections import namedtuple

import numpy as np
from sklearn.linear_model import LinearRegression
import xarray as xr

import multiLevelCoSurrogates as mlcs


AngleSummary = namedtuple('AngleSummary', 'alpha beta theta deg deg_low deg_high SSE df var_nhigh var_nlow')


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


def calc_angle(da: xr.DataArray, cost_ratio: float=None):
    """Calculate the global gradient angle of an Error Grid based
    on the slope of beta_1 / beta_2 from a linear regression fit.

    :param da:          xr.DataArray containing all model errors
    :param cost_ratio:  [optional] cost ratio between high- and low-fidelity
    """
    reg, SSE = mlcs.utils.error_grids.fit_lin_reg(da, calc_SSE=True)

    beta_high, beta_low = reg.coef_
    ratio = beta_high / beta_low
    if cost_ratio:
        ratio *= cost_ratio
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

    return AngleSummary(beta_high, beta_low, theta, mid_angle, min_angle, max_angle, SSE, df, var_nhigh, var_nlow)


class ConfidenceInterval(namedtuple('ConfidenceInterval', 'mean se lower upper')):

    def __contains__(self, value: float):
        return self.lower < value < self.upper

    def __str__(self):
        lower = self.lower if self.lower is not None else self.mean - 1.96*self.se
        upper = self.upper if self.upper is not None else self.mean + 1.96*self.se
        return f'95% CI: {self.mean:.4f} +/- {1.96*self.se:.4f} {np.array([lower, upper])}: ' \
               f'H0{" not" if 0 in self else ""} rejected'


class InvalidSlopeError(ValueError):
    """warns that the slope for determining 'tau' is negative in (at least) one axis"""


class UnhelpfulTauWarning(UserWarning):
    """warns that fidelity-selection parameter 'tau' is smaller than one"""


def calculate_tau(EG: xr.DataArray, cost_ratio):
    """Calculate slope ratio `tau` based on the angle of the EG's gradient"""
    reg = fit_lin_reg(EG)
    coef_high, coef_low = reg.coef_[:2]

    if coef_high < 0 or coef_low < 0:
        raise InvalidSlopeError('Error Grid implies improvement from removing samples')

    coef_low /= cost_ratio              # scale coef_low by cost ratio
    slope_ratio = coef_low / coef_high  # i.e. inverse of slope high/low
    tau = np.ceil(slope_ratio)

    if tau <= 1:
        warn('Low-fidelity not expected to add worthwhile information',
             category=UnhelpfulTauWarning)

    return tau
