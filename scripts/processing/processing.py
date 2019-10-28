#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
processing.py: Collection of data processing procedures that can be called
by explicit runner files.
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def plot_high_vs_low_num_samples(data, title, vmin=.5, vmax=100,
                                 points=(), save_as=None):
    norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    fig, ax = plt.subplots(figsize=(9,3.5))

    ax.set_aspect(1.)
    data = data.median(dim='rep')

    plt.title(f'Median MSE for high (hierarchical) model - {title}')
    img = ax.imshow(data.sel(model='high_hier'), cmap='viridis_r', norm=norm,
                    origin='lower', extent=get_extent(data))

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
    for p, style in zip(points, single_point_styles):
        ph, pl = tuple(map(int, p.DoE.split(':')))
        handle = ax.scatter(pl, ph, edgecolor='black', **style)
        pts.append((handle, f'{p.Author} ({p.Year}'))

    fig.colorbar(img, ax=ax, orientation='vertical')
    axy.set_ylabel('#High-fid samples')
    axx.set_xlabel('#Low-fid samples')

    if pts:
        handles, labels = zip(*pts)
        ax.legend(handles, labels, bbox_to_anchor=(1.2, 0.5), loc='center left')

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()


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


def plot_high_v_low_diff(to_plot, long_title, norm, save_as):
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
    plt.show()


def plot_t_scores(data, title, t_range=5, num_colors=10, save_as=None):
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
    plt.show()


def plot_extracts(data, title, save_as=None, show=False):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))

    n_highs = data.coords['n_high'].values
    n_lows = data.coords['n_low'].values
    for nhigh in range(np.min(n_highs), np.max(n_highs) + 1, 10):
        to_plot = data.sel(n_high=nhigh, model='high_hier').median(dim='rep')
        ax[0].plot(n_lows, to_plot, label=nhigh)
        ax[1].plot(n_lows, to_plot, label=nhigh)

    ax[0].set_title(title)
    ax[1].set_title(title + ' log-scale')
    ax[1].set_yscale('log')

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

