#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
processing.py: Collection of data processing procedures that can be called
by explicit runner files.
"""

from enum import IntEnum
from itertools import product
from collections import namedtuple
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from parse import Parser
from sklearn.linear_model import LinearRegression


__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


# defining some point styles
red_dot = {'marker': '.', 'color': 'red'}
blue_circle = {'marker': 'o', 'facecolors': 'none', 'color': 'blue'}

single_point_styles = [{'marker': m} for m in 'osHDPX*v^><']


LABEL_N_HIGH = "$n_h$"
LABEL_N_LOW = "$n_l$"

wide_figsize = (5.2, 2)
reg_figsize = (4, 2)


def get_extent(data: xr.DataArray):
    """Calculate an 'extent' for an Error Grid such that axis ticks are
    centered in the 'pixels'
    """
    return [
        np.min(data.n_low) - 0.5,
        np.max(data.n_low) - 0.5,
        np.min(data.n_high) - 0.5,
        np.max(data.n_high) - 0.5,
    ]


def full_extent(fig, ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles.
    Source:
    https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
    """
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    bbox = bbox.expanded(1.0 + pad, 1.0 + pad)

    return bbox.transformed(fig.dpi_scale_trans.inverted())


def plot_error_grid(data, title, vmin=.5, vmax=100, points=(),
                    contours=0, as_log=False, save_as=None,
                    show=False, include_comparisons=False,
                    include_colorbar=True, label_y=True):
    """Plot a heatmap of the median MSE for each possible combination of high
    and low-fidelity samples. For comparison, the MSE for the high-only and
    low-only models are displayed as a bar to the left and bottom respectively.

    :param data: `xr.DataArray` containing the MSE values
    :param title: title to use at top of the image
    :param vmin: minimum value for color scale normalization
    :param vmax: maximum value for color scale normalization
    :param points: iterable of namedtuples for fixed DoE's to plot
    :param contours: number of contour lines to draw. Default: 0
    :param as_log: display the log10 of the data or not (default False)
    :param save_as: desired filename for saving the image. Not saved if `None`
    :param show: whether or not to call `plt.show()`. Default: False
    :param include_comparisons: whether or not to include single-fidelity model
                                averages along axes. Default: False
    :param include_colorbar: whether or not to include a colorbar. Default: True
    :param label_y: whether or not to include axis label and ticks for y-axis. Default: True
    """
    if not (show or save_as):
        return  # no need to make the plot if not showing or saving it

    figsize = wide_figsize if include_comparisons else reg_figsize

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_aspect(1.)
    data = data.median(dim='rep')
    vmin = np.min(data) if vmin is None else vmin
    vmax = np.max(data) if vmax is None else vmax
    if as_log:
        data = np.log10(data)
        vmin = np.log10(vmin)
        vmax = np.log10(vmax)
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    else:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)

    extent = get_extent(data)
    imshow_style = {'cmap': 'viridis_r', 'norm': norm, 'origin': 'lower'}

    plt.title(fill(f'{"log10 " if as_log else ""}Median MSE for $z_h$ - {title}', width=32))

    da_hh = data.sel(model='high_hier')

    img = ax.imshow(da_hh, extent=extent, **imshow_style)
    if contours:
        ax.contour(da_hh, levels=contours, antialiased=False,
                   extent=extent, colors='black', alpha=.2, linewidths=1)

    divider = make_axes_locatable(ax)

    if include_comparisons:
        axx = divider.append_axes("bottom", size=.2, pad=0.05, sharex=ax)
        axy = divider.append_axes("left", size=.2, pad=0.05, sharey=ax)

        ax.xaxis.set_tick_params(labelbottom=False)
        ax.yaxis.set_tick_params(left=False, labelleft=False, which='both')
        axy.xaxis.set_tick_params(labelbottom=False)
        axx.yaxis.set_tick_params(left=False, labelleft=False, which='both')

        img = axy.imshow(data.sel(model='high').mean(dim='n_low').values.reshape(-1,1),
                         extent=(-0.5, 0.5, np.min(data.n_high)-.5, np.max(data.n_high)-.5),
                         **imshow_style)
        img = axx.imshow(data.sel(model='low').mean(dim='n_high').values.reshape(1,-1),
                         extent=(np.min(data.n_low)-.5, np.max(data.n_low)-.5, -0.5, 0.5),
                         **imshow_style)
        if label_y:
            axy.set_ylabel(LABEL_N_HIGH)
        else:
            axy.yaxis.set_tick_params(left=False, labelleft=False, which='both')
        axx.set_xlabel(LABEL_N_LOW)
    else:
        if label_y:
            ax.set_ylabel(LABEL_N_HIGH)
        else:
            ax.yaxis.set_tick_params(left=False, labelleft=False, which='both')
        ax.set_xlabel(LABEL_N_LOW)

    if include_colorbar:
        cax = divider.append_axes("right", size=0.2, pad=0.05)
        fig.colorbar(img, cax=cax)


    if points:
        pts = []
        for point, point_style in zip(points, single_point_styles):
            point_high, point_low = tuple(map(int, point.DoE.split(':')))
            if point_high <= np.max(data.n_high) and point_low <= np.max(data.n_low):
                handle = ax.scatter(point_low, point_high, edgecolor='black', **point_style)
                pts.append((handle, f'{point.Author} ({point.Year})'))

        handles, labels = zip(*pts)
        ax.legend(handles, labels, bbox_to_anchor=(1.2, 0.5), loc='center left')

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    if show:
        plt.show()
    plt.close('all')


def plot_multiple_error_grids(datas, titles, as_log=True,
                              vmin=None, vmax=None, contours=0,
                              save_as=None, show=False):
    """Plot a heatmap of the median MSE for each possible combination of high
    and low-fidelity samples. For comparison, the MSE for the high-only and
    low-only models are displayed as a bar to the left and bottom respectively.

    :param datas: `xr.DataArray`s containing the MSE values
    :param titles: titles to use at top of the image
    :param as_log: boolean to log-normalize the data or not
    :param vmin: minimum value for color scale normalization
    :param vmax: maximum value for color scale normalization
    :param contours: number of contour lines to draw. Default: 0
    :param save_as: desired filename for saving the image. Not saved if `None`
    :param show: whether or not to call `plt.show()`. Default: False
    """

    if not (show or save_as):
        return  # no need to make the plot if not showing or saving it

    ncols = len(datas)
    figsize = (4*ncols, 2)
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize)

    for ax, data, title in zip(axes, datas, titles):

        data = data.sel(model='high_hier').median(dim='rep')
        if as_log:
            data = np.log10(data)

        vmin = np.min(data) if not vmin else vmin
        vmax = np.max(data) if not vmax else vmax
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

        ax.set_aspect(1.)
        extent = get_extent(data)
        img = ax.imshow(data, cmap='viridis_r', norm=norm,
                        origin='lower', extent=extent)
        if contours:
            ax.contour(data, levels=contours, antialiased=False, extent=extent,
                       colors='black', alpha=.2, linewidths=1)

        ax.set_title(fill(f'{title}', width=32))
        ax.set_ylabel(LABEL_N_HIGH)
        ax.set_xlabel(LABEL_N_LOW)

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_error_grid_diff(data, title, max_diff=None, save_as=None):
    """Plot the difference between the high and hierarchical model
    """

    paired_diffs = data.sel(model='high') - data.sel(model='high_hier')
    to_plot = paired_diffs.median(dim='rep')
    if max_diff is None:
        max_diff = 2*min(abs(np.nanmin(to_plot)), np.nanmax(to_plot))

    norm = colors.SymLogNorm(linthresh=.01, vmin=-max_diff, vmax=max_diff, clip=True)
    long_title = f'Median of paired (high (hierarchical) - high (direct)) MSE - {title}'
    plot_high_v_low_diff(to_plot, long_title, norm, save_as)


def plot_inter_method_diff(data_A, data_B, name, model='high_hier',
                           max_diff=None, save_as=None):
    """Plot the difference between two Error Grids based on the same model
    """

    paired_diffs = data_A.sel(model=model) - data_B.sel(model=model)
    to_plot = paired_diffs.median(dim='rep')

    if max_diff is None:
        max_diff = 2*min(abs(np.nanmin(to_plot)), np.nanmax(to_plot))

    norm = colors.Normalize(vmin=-max_diff, vmax=max_diff, clip=True)
    long_title = f'high (hierarchical) MSE: {name}'
    plot_high_v_low_diff(to_plot, long_title, norm, save_as)


def plot_high_v_low_diff(to_plot, long_title, norm, save_as=None, show=False):
    """Plot a difference of Error Grids: using the pink-yellow-green colormap
    """
    if not (save_as or show):
        return  # no need to make the plot if not showing or saving it
    fig, ax = plt.subplots(figsize=wide_figsize)

    img = ax.imshow(to_plot, cmap='PiYG', norm=norm,
                    origin='lower', extent=get_extent(to_plot))
    fig.colorbar(img, ax=ax, orientation='vertical')
    ax.set_ylabel(LABEL_N_HIGH)
    ax.set_xlabel(LABEL_N_LOW)
    plt.title(long_title)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_t_scores(data: xr.DataArray, title: str, t_range: float=5, num_colors: int=10, save_as: str=None, show: bool=False):
    """Plot t-test scores of the difference between high and hierarchical model
    """
    if not (save_as or show):
        return  # no need to make the plot if not showing or saving it

    paired_differences = data.sel(model='high') - data.sel(model='high_hier')
    mean_paired_diff = paired_differences.mean(dim='rep')
    std_paired_diff = paired_differences.std(dim='rep', ddof=1)
    se_paired_diff = std_paired_diff / np.sqrt(data.shape[2])
    t_scores = mean_paired_diff / se_paired_diff

    norm = colors.Normalize(vmin=-t_range, vmax=t_range, clip=True)
    discrete_cmap = plt.get_cmap('PuOr_r', num_colors)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    img = ax.imshow(t_scores, cmap=discrete_cmap, norm=norm,
                    origin='lower', extent=get_extent(data))
    fig.colorbar(img, ax=ax, orientation='vertical', ticks=np.arange(-t_range, t_range, 1))
    ax.set_title(f"Paired difference t-scores - {title}")

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_extracts(data: xr.DataArray, title: str, save_as: str=None, show: bool=False, *,
                  normalize: bool=False, max_x: int=None):
    """Plot extracts from an Error Grid, i.e. the MSE versus number of low-fidelity
    samples given a fixed number of high-fidelity samples
    """

    if not (save_as or show):
        return  # no need to make the plot if not showing or saving it
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    n_highs = data.coords['n_high'].values
    for nhigh in range(np.min(n_highs), np.max(n_highs) + 1, 10):
        to_plot = data.sel(n_high=nhigh, model='high_hier').median(dim='rep').dropna(dim='n_low')
        n_lows = to_plot.coords['n_low'].values

        if normalize:
            x = n_lows/nhigh
            to_plot /= to_plot.sel(n_low=nhigh+1)
        else:
            x = n_lows

        axes[0].plot(x, to_plot, label=nhigh, zorder=nhigh)
        axes[1].plot(x, to_plot, label=nhigh, zorder=nhigh)

    axes[0].set_title(title)
    axes[1].set_title(title + ' log-scale')
    axes[1].set_yscale('log')

    if normalize:
        x_label = "Ratio high-fidelity/low-fidelity samples"
        y_label = "Relative Median MSE"
    else:
        x_label = "Number of low-fidelity samples"
        y_label = "Median MSE"

    for ax in axes:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(left=0, right=max_x)

    plt.legend(loc=0)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_multi_file_extracts(data_arrays, title: str, save_as: str=None, show: bool=False):
    """Plot Error Grid extracts drawn from multiple files
    """
    if not (save_as or show) or not data_arrays:
        return

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    alphas = 1 - np.linspace(0, 1, len(data_arrays), endpoint=False)

    for data, alpha in zip(data_arrays, alphas):
        data.load()
        n_highs = data.coords['n_high'].values
        n_lows = data.coords['n_low'].values
        for i, nhigh in enumerate(range(np.min(n_highs), np.max(n_highs) + 1, 10)):
            to_plot = data.sel(n_high=nhigh, model='high_hier').median(dim='rep')
            ax[0].plot(n_lows, to_plot, color=f'C{i}', alpha=alpha)
            ax[1].plot(n_lows, to_plot, color=f'C{i}', alpha=alpha)
        data.close()

    ax[0].set_title(title)
    ax[1].set_title(title + ' log-scale')
    ax[1].set_yscale('log')

    legend_input = [(Line2D([0], [0], color=f'C{i}'), nhigh)
                    for i, nhigh in enumerate(range(np.min(n_highs), np.max(n_highs) + 1, 10))]

    plt.legend(*zip(*legend_input), loc=0)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def fit_lin_reg(da: xr.DataArray, calc_SSE: bool=False):
    """Return lin-reg coefficients after training index -> value"""

    series = da.to_series().dropna()
    X = np.array(series.index.tolist())[:,:2]  # remove rep_idx (3rd column)
    y = np.log10(series.values)
    reg = LinearRegression().fit(X, y)

    if not calc_SSE:
        return reg

    pred_y = reg.predict(X)
    SSE = np.sum((pred_y - y)**2)
    return reg, SSE


class ConfidenceInterval(namedtuple('ConfidenceInterval', 'mean se lower upper')):

    def __contains__(self, value: float):
        return self.lower < value < self.upper

    def __str__(self):
        lower = self.lower if self.lower is not None else self.mean - 1.96*self.se
        upper = self.upper if self.upper is not None else self.mean + 1.96*self.se
        return f'95% CI: {self.mean:.4f} +/- {1.96*self.se:.4f} {np.array([lower, upper])}: ' \
               f'H0{" not" if 0 in self else ""} rejected'


def calc_angle(da: xr.DataArray):
    """Calculate the global gradient angle of an Error Grid based
    on the slope of beta_1 / beta_2 from a linear regression fit.
    """
    AngleSummary = namedtuple('AngleSummary', 'alpha beta theta deg deg_low deg_high')
    reg, SSE = fit_lin_reg(da, calc_SSE=True)

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


def get_gradient_angles(directory: Path, force_regen: bool=False):
    """Get a Pandas DataFrame of the gradient angles for all .nc files in the
    given directory. If calculated before, they are loaded from a summary file
    'gradients.csv', unless `force_regen` is set to `True`, in which case they
    will be recalculated.

    :param directory:   Directory with files to be summarized
    :param force_regen: (Re)generate summary file, even if it already exists
    """
    gradients_filename = directory / 'gradients.csv'
    if force_regen or not gradients_filename.exists():
        df = calculate_gradient_angles(directory)
        df['param'] = df['param'].astype('float64')
        df.to_csv(gradients_filename, index=False)
    else:
        df = pd.read_csv(gradients_filename)
    return df


def calculate_gradient_angles(directory: Path):
    """Given a directory, calculate the gradient angle for each .nc file"""

    Record = namedtuple(
        'Record',
        'surrogate category fname ndim param alpha beta theta deg deg_low deg_high'
    )
    records = []
    adjustable_parser = Parser("{surrogate:w}-{ndim:d}d-Adjustable-{fname}-{param:f}.nc")
    regular_parser = Parser("{surrogate:w}-{ndim:d}d-{fname}.nc")
    for file in sorted(directory.iterdir()):
        if 'Adjustable' in file.name:
            match = adjustable_parser.parse(file.name)
            category = 'adjustable'
        else:
            match = regular_parser.parse(file.name)
            category = 'regular'

        if not match:
            continue
        param = match['param'] if 'param' in match else None

        with xr.open_dataset(file) as ds:
            da = ds['mses'].sel(model='high_hier')
        with da.load() as da:
            angle_summary = calc_angle(da)
        records.append(
            Record(match['surrogate'], category, match['fname'].lower(), match['ndim'], param,
                   *angle_summary)
        )
    return pd.DataFrame.from_records(records, columns=Record._fields)


class Comparison(IntEnum):
    NO_MATCH = 0
    CI_MATCH = 1
    SINGLE_MATCH = 2
    DOUBLE_MATCH = 3


def determine_match(CI1, CI2):
    """Determine the kind of overlap between two ConfidenceIntervals"""
    # is the midpoint of one CI within the bounds of the other CI?
    covered_1 = CI1.mean in CI2
    covered_2 = CI2.mean in CI1

    if covered_1 and covered_2:
        return Comparison.DOUBLE_MATCH
    if covered_1 or covered_2:
        return Comparison.SINGLE_MATCH
    if CI1.lower in CI2 or CI1.upper in CI2:  # reverse is implied
        return Comparison.CI_MATCH
    return Comparison.NO_MATCH
