#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
processing.py: Collection of data processing procedures that can be called
by explicit runner files.
"""

from itertools import product
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LinearRegression


__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


# defining some point styles
red_dot = {'marker': '.', 'color': 'red'}
blue_circle = {'marker': 'o', 'facecolors': 'none', 'color': 'blue'}

single_point_styles = [{'marker': m}  # 'fillstyle': f,
                       for m in 'osHDPX*v^><']
                       #for f, m in product(['full', None], 'oSHDPX*v^><')]


def get_extent(data):
    return [
        np.min(data.n_low) - 0.5,
        np.max(data.n_low) - 0.5,
        np.min(data.n_high) - 0.5,
        np.max(data.n_high) - 0.5,
    ]


def plot_high_vs_low_num_samples(data, title, vmin=.5, vmax=100, points=(),
                                 contours=0, as_log=False, save_as=None, show=False):
    """Plot a heatmap of the median MSE for each possible combination of high
    and low-fidelity samples. For comparison, the MSE for the high-only and
    low-only models are displayed as a bar to the left and bottom respectively.

    :param data: `xr.DataArray` containing the MSE values
    :param title: title to use at top of the image
    :param vmin: minimum value for colorscale normalization
    :param vmax: maximum value for colorscale normalization
    :param points: iterable of namedtuples for fixed DoE's to plot
    :param contours: number of contour lines to draw. Default: 0
    :param as_log: display the log10 of the data or not (default False)
    :param save_as: desired filename for saving the image. Not saved if `None`
    :param show: whether or not to call `plt.show()`. Default: False
    :return:
    """
    if not (show or save_as):
        return  # no need to make the plot if not showing or saving it

    fig, ax = plt.subplots(figsize=(9,3.5))

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
    plt.title(f'Median MSE for high (hierarchical) model - {title}')
    img = ax.imshow(data.sel(model='high_hier'), cmap='viridis_r', norm=norm,
                    origin='lower', extent=extent)
    if contours:
        ax.contour(data.sel(model='high_hier'), levels=contours, antialiased=False,
                   extent=extent, colors='black', alpha=.2, linewidths=1)

    divider = make_axes_locatable(ax)
    axx = divider.append_axes("bottom", size=.2, pad=0.05, sharex=ax)
    axy = divider.append_axes("left", size=.2, pad=0.05, sharey=ax)

    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    axy.xaxis.set_tick_params(labelbottom=False)
    axx.yaxis.set_tick_params(labelleft=False)

    img = axy.imshow(data.sel(model='high').mean(dim='n_low').values.reshape(-1,1),
                     cmap='viridis_r', norm=norm, origin='lower',
                     extent=(-0.5, 0.5, np.min(data.n_high)-.5, np.max(data.n_high)-.5))
    img = axx.imshow(data.sel(model='low').mean(dim='n_high').values.reshape(1,-1),
                     cmap='viridis_r', norm=norm, origin='lower',
                     extent=(np.min(data.n_low)-.5, np.max(data.n_low)-.5, -0.5, 0.5))

    pts = []
    for point, style in zip(points, single_point_styles):
        point_high, point_low = tuple(map(int, point.DoE.split(':')))
        if point_high <= np.max(data.n_high) and point_low <= np.max(data.n_low):
            handle = ax.scatter(point_low, point_high, edgecolor='black', **style)
            pts.append((handle, f'{point.Author} ({point.Year})'))

    fig.colorbar(img, ax=ax, orientation='vertical')
    axy.set_ylabel('#High-fid samples')
    axx.set_xlabel('#Low-fid samples')

    if pts:
        handles, labels = zip(*pts)
        ax.legend(handles, labels, bbox_to_anchor=(1.2, 0.5), loc='center left')

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close()


def plot_multiple_high_vs_low_num_samples(datas, titles, as_log=True,
                                          vmin=None, vmax=None, contours=0,
                                          save_as=None, show=False):
    """Plot a heatmap of the median MSE for each possible combination of high
    and low-fidelity samples. For comparison, the MSE for the high-only and
    low-only models are displayed as a bar to the left and bottom respectively.

    :param datas: `xr.DataArray`s containing the MSE values
    :param titles: titles to use at top of the image
    :param as_log: boolean to log-normalize the data or not
    :param vmin: minimum value for colorscale normalization
    :param vmax: maximum value for colorscale normalization
    :param contours: number of contour lines to draw. Default: 0
    :param save_as: desired filename for saving the image. Not saved if `None`
    :param show: whether or not to call `plt.show()`. Default: False
    :return:
    """
    if not (show or save_as):
        return  # no need to make the plot if not showing or saving it

    ncols = len(datas)
    figsize = (6.5*ncols, 3.25)
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

        ax.set_title(f'{title}')
        ax.set_ylabel('#High-fid samples')
        ax.set_xlabel('#Low-fid samples')

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close()


def plot_high_vs_low_num_samples_diff(data, title, max_diff=None, save_as=None):

    paired_diffs = data.sel(model='high') - data.sel(model='high_hier')
    to_plot = paired_diffs.median(dim='rep')
    if max_diff is None:
        max_diff = 2*min(abs(np.nanmin(to_plot)), np.nanmax(to_plot))

    norm = colors.SymLogNorm(linthresh=.01, vmin=-max_diff, vmax=max_diff, clip=True)

    long_title = f'Median of paired (high (hierarchical) - high (direct)) MSE - {title}'

    plot_high_v_low_diff(to_plot, long_title, norm, save_as)


def plot_inter_method_diff(data_A, data_B, name, model='high_hier',
                           max_diff=None, save_as=None):

    paired_diffs = data_A.sel(model=model) - data_B.sel(model=model)
    to_plot = paired_diffs.median(dim='rep')

    if max_diff is None:
        max_diff = 2*min(abs(np.nanmin(to_plot)), np.nanmax(to_plot))

    norm = colors.Normalize(vmin=-max_diff, vmax=max_diff, clip=True)

    long_title = f'high (hierarchical) MSE: {name}'

    plot_high_v_low_diff(to_plot, long_title, norm, save_as)


def plot_high_v_low_diff(to_plot, long_title, norm, save_as=None, show=False):
    if not (save_as or show):
        return  # no need to make the plot if not showing or saving it
    fig, ax = plt.subplots(figsize=(9, 3.5))

    img = ax.imshow(to_plot, cmap='PiYG', norm=norm,
                    origin='lower', extent=get_extent(to_plot))
    fig.colorbar(img, ax=ax, orientation='vertical')
    ax.set_ylabel('#High-fid samples')
    ax.set_xlabel('#Low-fid samples')
    plt.title(long_title)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close()


def plot_t_scores(data, title, t_range=5, num_colors=10, save_as=None, show=False):
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
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close()


def plot_extracts(data, title, save_as=None, show=False, *, normalize=False, max_x=None):
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
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close()


def plot_multi_file_extracts(data_arrays, title, save_as=None, show=False):
    if not (save_as or show) or not data_arrays:
        return

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))

    for idx, data in enumerate(data_arrays):
        alpha = 1 - (idx/len(data_arrays))
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
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close()


def fit_lin_reg(da: xr.DataArray):
    """Return lin-reg coefficients after training index -> value"""

    # clean data
    series = da.to_series().dropna()

    # extract X and y
    X = np.array(series.index.tolist())
    y = np.log10(series.values)

    # compute and return coefficients
    reg = LinearRegression().fit(X, y)

    return reg

    ### calculate statistics ###
    # mean = np.mean(y)
    # TSS = np.sum((y-mean) ** 2)
    # SSE = reg._residues
    # r_2 = (TSS-SSE)/TSS
    # t_score = np.sqrt(r_2) / np.sqrt((1-r_2) / (len(y)-2))

    # return results nicely
    # Coefficients = namedtuple('Coefficients', series.index.names)
    # return Coefficients(*reg.coef_)
