#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
processing.py: Collection of data processing procedures that can be called
by explicit runner files.
"""

from enum import IntEnum
from collections import namedtuple
from operator import itemgetter
from pathlib import Path
from textwrap import fill
from typing import Union

import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import mf2
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from parse import Parser

import multiLevelCoSurrogates as mlcs

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

mpl.rcParams['savefig.dpi'] = 300

# defining some point styles
red_dot = {'marker': '.', 'color': 'red'}
blue_circle = {'marker': 'o', 'facecolors': 'none', 'color': 'blue'}

single_point_styles = [{'marker': m} for m in 'osHDPX*v^><']


LABEL_N_HIGH = "$n_h$"
LABEL_N_LOW = "$n_l$"

wide_figsize = (5.2, 2)
reg_figsize = (4, 2)

suffixes = ['.pdf', '.png']


def get_extent(data: xr.DataArray):
    """Calculate an 'extent' for an Error Grid such that axis ticks are
    centered in the 'pixels'
    """
    return [
        np.min(data.n_low).item() - 0.5,
        np.max(data.n_low).item() + 0.5,
        np.min(data.n_high).item() - 0.5,
        np.max(data.n_high).item() + 0.5,
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


def plot_archive(
        archive: mlcs.CandidateArchive,
        func: mf2.MultiFidelityFunction,
        title: str,
        save_as: Union[str, Path],
        suffixes=('.pdf', '.png'),
) -> None:
    """Plot given archive using parameters of func

    param archive: CandidateArchive to plot
    param func:    MultiFidelityFunction to query for name and bounds
    param title:   Title of the plot
    param save_as: Filename to save as
    """
    if func.ndim != 2:
        raise NotImplementedError("plotting for other than 2D not implemented")

    num_intervals = 50
    u_bound, l_bound = func.u_bound, func.l_bound
    step = (u_bound - l_bound) / num_intervals
    surf = mlcs.createsurface(func.high, l_bound=l_bound, u_bound=u_bound, step=step, wide=False)
    extent = [
        l_bound[0] - step[0]/2,
        u_bound[0] + step[0]/2,
        l_bound[1] - step[1]/2,
        u_bound[1] + step[1]/2
    ]

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.imshow(surf.Z, extent=extent, cmap='viridis_r', origin='lower')
    for fid, style in zip(['high', 'low'], [red_dot, blue_circle]):
        points = archive.getcandidates(fid).candidates
        ax.scatter(*points.T, **style, label=f'{fid}-fidelity samples')
    ax.legend(loc=0)
    for suffix in suffixes:
        fig.savefig(save_as.with_suffix(suffix))
    fig.clear()
    plt.close('all')


def plot_error_grid(data, title, vmin=.5, vmax=100, points=(),
                    contours=0, as_log=False, save_as=None,
                    show=False, include_comparisons=False, gradient_arrow=False,
                    include_colorbar=True, label_y=True, title_width=None,
                    xlim=None, ylim=None,):
    """Plot a heatmap of the median MSE for each possible combination of high
    and low-fidelity samples. For comparison, the MSE for the high-only and
    low-only models are displayed as a bar to the left and bottom respectively.

    :param data:                `xr.DataArray` containing the MSE values
    :param title:               title to use at top of the image
    :param vmin:                minimum value for color scale normalization
    :param vmax:                maximum value for color scale normalization
    :param points:              iterable of namedtuples for fixed DoE's to plot
    :param contours:            number of contour lines to draw. Default: 0
    :param as_log:              display the log10 of the data or not (default False)
    :param save_as:             desired filename for saving the image. Not saved if `None`
    :param show:                whether or not to call `plt.show()`. Default: False
    :param include_comparisons: whether or not to include single-fidelity model
                                averages along axes. Default: False
    :param include_colorbar:    whether or not to include a colorbar. Default: True
    :param label_y:             whether or not to include axis label and ticks for y-axis. Default: True
    :param gradient_arrow:      whether or not to add an arrow indicating gradient direction through
                                the center of the figure. Default: False
    :param title_width:         maximum width of the title for line wrapping
    :param xlim:                base x-limits, upper will extend to fit data
    :param ylim:                base y-limits, upper will extend to fit data
    """
    if not (show or save_as):
        return  # no need to make the plot if not showing or saving it

    figsize = wide_figsize if include_comparisons else reg_figsize

    fig, ax = plt.subplots(figsize=figsize)

    if gradient_arrow:
        add_gradient_arrow_line_to_axis(data.sel(model='high_hier'), ax)

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

    plot_title = f'{"log10 " if as_log else ""}Median MSE for $z_h$ - {title}'
    if title_width:
        plot_title = fill(plot_title, width=title_width)
    plt.title(plot_title)

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

    if xlim:
        ax.set_xlim((xlim[0]-.5, max(xlim[1], max(data.n_low))+.5))
    if ylim:
        ax.set_ylim((ylim[0]-.5, max(ylim[1], max(data.n_high))+.5))

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
        for suffix in suffixes:
            plt.savefig(f'{save_as}{suffix}', bbox_inches='tight')
    if show:
        plt.show()
    fig.clear()
    plt.close('all')


def plot_multiple_error_grids(datas, titles, as_log=True, gradient_arrow=False,
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

    if not show and not save_as:
        return  # no need to make the plot if not showing or saving it

    ncols = len(datas)
    figsize = (3*ncols, 2)
    fig, axes = plt.subplots(ncols=ncols, figsize=figsize, sharey=True)

    is_first_ax = True
    for ax, data, title in zip(axes, datas, titles):

        if gradient_arrow:
            add_gradient_arrow_line_to_axis(data.sel(model='high_hier'), ax)

        data = data.sel(model='high_hier').median(dim='rep')
        if as_log:
            data = np.log10(data)

        vmin = vmin or np.min(data)
        vmax = vmax or np.max(data)
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

        ax.set_aspect(1.)
        extent = get_extent(data)
        img = ax.imshow(data, cmap='viridis_r', norm=norm,
                        origin='lower', extent=extent)
        if contours:
            ax.contour(data, levels=contours, antialiased=False, extent=extent,
                       colors='black', alpha=.2, linewidths=1)

        ax.set_title(fill(f'{title}', width=32))
        ax.set_xlabel(LABEL_N_LOW)
        if is_first_ax:
            ax.set_ylabel(LABEL_N_HIGH)
            is_first_ax = False
        else:
            ax.yaxis.set_tick_params(left=False, labelleft=False, which='both')

    plt.tight_layout()
    if save_as:
        for suffix in suffixes:
            plt.savefig(f'{save_as}{suffix}', bbox_inches='tight')
    if show:
        plt.show()
    fig.clear()
    plt.close('all')


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
        for suffix in suffixes:
            plt.savefig(f'{save_as}{suffix}', bbox_inches='tight')
    if show:
        plt.show()
    fig.clear()
    plt.close('all')


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
        for suffix in suffixes:
            plt.savefig(f'{save_as}{suffix}', bbox_inches='tight')
    if show:
        plt.show()
    fig.clear()
    plt.close('all')


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
    axes[1].set_title(f'{title} log-scale')
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
        for suffix in suffixes:
            plt.savefig(f'{save_as}{suffix}', bbox_inches='tight')
    if show:
        plt.show()
    fig.clear()
    plt.close('all')


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
    ax[1].set_title(f'{title} log-scale')
    ax[1].set_yscale('log')

    legend_input = [(Line2D([0], [0], color=f'C{i}'), nhigh)
                    for i, nhigh in enumerate(range(np.min(n_highs), np.max(n_highs) + 1, 10))]

    plt.legend(*zip(*legend_input), loc=0)
    plt.tight_layout()

    if save_as:
        for suffix in suffixes:
            plt.savefig(f'{save_as}{suffix}', bbox_inches='tight')
    if show:
        plt.show()
    fig.clear()
    plt.close('all')


def add_gradient_arrow_line_to_axis(da: xr.DataArray, ax: plt.Axes):
    """Add a line with arrow to axis `ax` to indicate gradient direction"""
    # preparing variables for n_h = a*n_l + b
    reg = mlcs.utils.error_grids.fit_lin_reg(da)
    n_l_min, n_l_max, n_h_min, n_h_max = get_extent(da)
    a = np.divide(*reg.coef_)
    b = (n_h_max/2) - a*(n_l_max/2)

    if np.isinf(a):  # 90 degrees, vertical
        coords = np.array([
            [n_l_max/2, n_h_min],
            [n_l_max/2, n_h_max],
        ])
    elif b == 0:  # crossing the origin
        coords = np.array([
            [n_l_min, n_h_min],
            [n_l_max, n_h_max],
        ])
    elif a < 0 or b < n_h_min:  # crossing x-axis in plotted area
        coords = np.array([
            [         (-b)/a, n_h_min],
            [(n_h_max - b)/a, n_h_max],
        ])
    else:  # b > n_h_min; crossing y-axis
        coords = np.array([
            [n_l_min,               b],
            [n_l_max, (a*n_l_max) + b],
        ])

    mid_point = np.sum(coords, axis=0) / 2

    ax.annotate('', xytext=coords[0], xy=mid_point, arrowprops={'arrowstyle': '->, head_length=.8, head_width=.4', 'shrinkA': 2.5, 'shrinkB': 0})
    ax.plot(coords.T[0], coords.T[1], color='black')


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
            angle_summary = mlcs.utils.error_grids.calc_angle(da)
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


def gifify_in_folder(in_folder: Path, base_name: str):
    """Combine all images matching `{base_name}_{idx}.png` in `in_folder` into `base_name.gif`"""

    template = Parser(base_name + '_{idx:d}.png')
    files_to_gifify = sorted([
        (file, match['idx'])
        for file in in_folder.iterdir()
        if (match := template.parse(file.name))
    ], key=itemgetter(1))

    with imageio.get_writer(in_folder / f"{base_name}.gif", mode='I', duration=0.5) as writer:
        for (file, _) in files_to_gifify:
            writer.append_data(imageio.imread(file))
