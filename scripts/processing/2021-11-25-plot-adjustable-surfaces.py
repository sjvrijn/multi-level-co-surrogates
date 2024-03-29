#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
2021-11-25-plot-adjustable-surfaces.py: Some initial plotting of data gathered in .csv/.pkl
files during the simple-mfbo runs
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import mf2
import parse
from pyprojroot import here

import processing as proc

module_path = str(here())
if module_path not in sys.path:
    sys.path.append(module_path)

import multiLevelCoSurrogates as mlcs


print(f'Running script: {__file__}')


plot_path = here('plots/2021-11-25-plot-adjustable-surfaces/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)


def plot_adjustable_surface_collection(func, fig_size, params=None):
    """Plot all relevant surfaces for an adjustable multi-fidelity function.

    Typically, this is the high-fidelity surface, and the low-fidelity surfaces
    for A=[0.00, 0.05, 0.10, ..., 0.95, 1.00] (i.e. linspace(0, 1, 21))

    :param func:     Function to plot
    :param fig_size: Size of the figure in inches. Default: 3
    :param params:   Set of parameter values to plot for. Default: linspace(0, 1, 21)

    """

    if not params:
        params = np.round(np.linspace(0, 1.0, 21), 2)
    elif isinstance(params, int):
        params = np.round(np.linspace(0, 1.0, params), 2)

    #gather surfaces, titles and filenames
    to_plot = []

    #plot high-fid
    f = func(0)
    steps = (f.u_bound - f.l_bound) / 25
    print(f.name)
    base_name = parse.parse('Adjustable {name} 0', f.name)['name']

    for suffix in proc.extensions:
        filename = plot_path / f'adjustable-{base_name.lower()}-high-fid.{suffix}'
        surface = mlcs.plotting.createsurface(f.high, l_bound=f.l_bound, u_bound=f.u_bound, step=steps)
        title = f'{base_name}: high-fidelity'
        to_plot.append((surface, title, filename))

    #plot low-fid
    for p in params:
        f = func(p)
        p_str = f'{p:.2f}'
        for suffix in proc.extensions:
            filename = plot_path / f'adjustable-{base_name.lower()}-{p_str.replace(".", "")}-low-fid.{suffix}'
            surface = mlcs.plotting.createsurface(f.low, l_bound=f.l_bound, u_bound=f.u_bound, step=steps)
            title = f'{base_name} low-fidelity: A={p:.2f}'
            to_plot.append((surface, title, filename))

    # Determine range of surface.Z values for axis-limit or colormap range
    if 'branin' in f.name.lower():
        zlim = [-200, 400]
        zticks = [-200, 0, 200, 400]
    elif 'paciorek' in f.name.lower():
        zlim = [-10, 10]
        zticks = [-10, -5, 0, 5, 10]
    else:
        z_mins, z_maxs = zip(*[(np.min(s.Z), np.max(s.Z)) for s, _, _ in to_plot])
        zlim = [min(z_mins), max(z_maxs)]
        zticks = None

    figsize = (fig_size, 0.9*fig_size)
    for surface, title, filename in to_plot:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': '3d'})
        mlcs.plotting.plotsurfaceonaxis(ax, surface, title, plot_type='wireframe', contour=False)
        ax.set_zlim(zlim)
        if zticks:
            ax.set_zticks(zticks)
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--figsize', type=float, default=3)
    parser.add_argument('--params', nargs='*', type=float)
    args = parser.parse_args()

    functions = [
        mf2.adjustable.branin,
        mf2.adjustable.paciorek,
    ]

    for function in functions:
        plot_adjustable_surface_collection(function, args.figsize, args.params)

