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
from pyprojroot import here

import processing as proc

module_path = str(here())
if module_path not in sys.path:
    sys.path.append(module_path)

import multiLevelCoSurrogates as mlcs


print(f'Running script: {__file__}')


plot_path = here('plots/2021-11-25-plot-adjustable-surfaces/', warn=False)
plot_path.mkdir(exist_ok=True, parents=True)


def plot_adjustable_surface_collection(func: mf2.MultiFidelityFunction, params=None):
    """Plot all relevant surfaces for an adjustable multi-fidelity function.

    Typically, this is the high-fidelity surface, and the low-fidelity surfaces
    for A=[0.00, 0.05, 0.10, ..., 0.95, 1.00] (i.e. linspace(0, 1, 21))

    :param func:   Function to plot
    :param params: Set of parameter values to plot for. Default: linspace(0, 1, 21)

    """

    if params is None:
        params = np.round(np.linspace(0, 1.0, 21), 2)
    elif isinstance(params, int):
        params = np.round(np.linspace(0, 1.0, params), 2)

    #gather surfaces, titles and filenames
    to_plot = []

    #plot high-fid
    f = func(0)
    steps = (f.u_bound - f.l_bound) / 25
    print(f.name)
    for suffix in proc.extensions:
        filename = plot_path / f'{f.name[:-2].lower().replace(" ", "_")}-high-fid.{suffix}'
        surface = mlcs.plotting.createsurface(f.high, l_bound=f.l_bound, u_bound=f.u_bound, step=steps)
        #mlcs.plotting.plotsurfaces([surface], titles=[f.name[:-2]], save_as=filename, as_3d=True, show=False)
        to_plot.append((surface, f'{f.name[:-2]}: high-fidelity', filename))

    #plot low-fid
    for f in (func(p) for p in params):
        for suffix in proc.extensions:
            filename = plot_path / f'{f.name.lower().replace(" ", "_").replace(".", "")}-low-fid.{suffix}'
            surface = mlcs.plotting.createsurface(f.low, l_bound=f.l_bound, u_bound=f.u_bound, step=steps)
            #mlcs.plotting.plotsurfaces([surface], titles=[f.name], save_as=filename, as_3d=True, show=False)
            to_plot.append((surface, f.name, filename))

    # Determine range of surface.Z values for axis-limit or colormap range
    if 'branin' in f.name.lower():
        zlim = [-180, 450]
    elif 'paciorek' in f.name.lower():
        zlim = [-9, 9]
    else:
        z_mins, z_maxs = zip(*[(np.min(s.Z), np.max(s.Z)) for s, _, _ in to_plot])
        zlim = [min(z_mins), max(z_maxs)]


    for surface, title, filename in to_plot:
        fig, ax = plt.subplots(figsize=(5, 4.5), subplot_kw={'projection': '3d'})
        mlcs.plotting.plotsurfaceonaxis(ax, surface, title, plot_type='wireframe', contour=False)
        ax.set_zlim(zlim)
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-regen', action='store_true')
    args = parser.parse_args()

    functions = [
        mf2.adjustable.branin,
        mf2.adjustable.paciorek,
    ]

    for function in functions:
        plot_adjustable_surface_collection(function)

