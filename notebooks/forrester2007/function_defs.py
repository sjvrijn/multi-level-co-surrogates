import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import clear_output
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyDOE import lhs
from pyprojroot import here

module_path = str(here())
if module_path not in sys.path:
    sys.path.append(module_path)

import multiLevelCoSurrogates as mlcs



def low_random_sample(ndim, nlow):
    return np.random.rand(nlow, ndim)

def low_lhs_sample(ndim, nlow):
    if ndim == 1:
        return np.linspace(0,1,nlow).reshape(-1,1)
    elif ndim > 1:
        return lhs(ndim, nlow)


def create_mse_tracking(func, sample_generator,
                        max_high=40, max_low=100, num_reps=30,
                        min_high=2, min_low=3):
    ndim = func.ndim
    mse_tracking = np.empty((max_high+1, max_low+1, num_reps, 3))
    mse_tracking[:] = np.nan
    cases = list(product(range(min_high, max_high+1), range(min_low, max_low+1), range(num_reps)))

    for idx, case in enumerate(cases):
        num_high, num_low, rep = case

        if num_high >= num_low:
            continue
        if idx % 100 == 0:
            clear_output()
            print(f'{idx}/{len(cases)}')

        low_x = sample_generator(ndim, num_low)
        high_x = low_x[np.random.choice(num_low, num_high, replace=False)]

        archive = mlcs.CandidateArchive(ndim=ndim, fidelities=['high', 'low', 'high-low'])
        archive.add_candidates(low_x, func.low(low_x), fidelity='low')
        archive.add_candidates(high_x, func.high(high_x), fidelity='high')

        mfbo = mlcs.MultiFidelityBO(func, archive)
        mse_tracking[num_high, num_low, rep] = mfbo.getMSE()

    clear_output()
    print(f'{len(cases)}/{len(cases)}')
    return mse_tracking


def plot_high_vs_low_num_samples(data, title, vmin=.5, vmax=100, save_as=None):
    norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    fig, ax = plt.subplots(figsize=(9,3.5))

    ax.set_aspect(1.)
    data = data.median(dim='rep')

    plt.title(f'Median MSE for high (hierarchical) model - {title}')
    img = ax.imshow(data.sel(model='high_hier'), cmap='viridis_r', norm=norm, origin='lower')

    divider = make_axes_locatable(ax)
    axx = divider.append_axes("bottom", size=.2, pad=0.05, sharex=ax)
    axy = divider.append_axes("left", size=.2, pad=0.05, sharey=ax)

    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    axy.xaxis.set_tick_params(labelbottom=False)
    axx.yaxis.set_tick_params(labelleft=False)

    img = axy.imshow(data.sel(model='high').mean(dim='n_low').values.reshape(-1,1), cmap='viridis_r', norm=norm, origin='lower')
    img = axx.imshow(data.sel(model='low').mean(dim='n_high').values.reshape(1,-1), cmap='viridis_r', norm=norm, origin='lower')

    fig.colorbar(img, ax=ax, orientation='vertical')
    axy.set_ylabel('#High-fid samples')
    axx.set_xlabel('#Low-fid samples')

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

    plot_high_v_low(long_title, norm, save_as, to_plot)


def plot_inter_method_diff(data_A, data_B, name, max_diff=None, save_as=None):
    to_plot = np.nanmedian(data_A.sel(model='high_hier') - data_B.sel(model='high_hier'), axis=2)

    if max_diff is None:
        max_diff = 2*min(abs(np.nanmin(to_plot)), np.nanmax(to_plot))

    norm = colors.Normalize(vmin=-max_diff, vmax=max_diff, clip=True)

    long_title = f'high (hierarchical) MSE: {name}'

    plot_high_v_low(long_title, norm, save_as, to_plot)


def plot_high_v_low(long_title, norm, save_as, to_plot):
    fig, ax = plt.subplots(figsize=(9, 3.5))
    img = ax.imshow(to_plot, cmap='RdYlGn', norm=norm, origin='lower')
    fig.colorbar(img, ax=ax, orientation='vertical')
    ax.set_ylabel('#High-fid samples')
    ax.set_xlabel('#Low-fid samples')
    plt.title(long_title)
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)
    plt.show()


# defining some point styles
red_dot = {'marker': '.', 'color': 'red'}
blue_circle = {'marker': 'o', 'facecolors': 'none', 'color': 'blue'}



def create_models_and_compare(func, low, high, steps=None, save_as=None):
    archive = mlcs.CandidateArchive(ndim=2, fidelities=['high', 'low', 'high-low'])
    archive.add_candidates(low, func.low(low), fidelity='low')
    archive.add_candidates(high, func.high(high), fidelity='high')

    mfbo = mlcs.MultiFidelityBO(func, archive, schema=[1,1])

    surf_high = mlcs.createsurface(func.high, u_bound=func.u_bound, l_bound=func.l_bound, step=steps)
    surf_low = mlcs.createsurface(func.low, u_bound=func.u_bound, l_bound=func.l_bound, step=steps)

    surf_high_model = mlcs.createsurface(mfbo.models['high'].predict, u_bound=func.u_bound, l_bound=func.l_bound, step=steps)
    surf_low_model = mlcs.createsurface(mfbo.models['low'].predict, u_bound=func.u_bound, l_bound=func.l_bound, step=steps)

    points_high = [mlcs.ScatterPoints(*archive.get_candidates(fidelity='high'), red_dot)]
    points_low = [mlcs.ScatterPoints(*archive.get_candidates(fidelity='low'), blue_circle)]

    points = [
        points_high, points_low,
        points_high, points_low,
    ]

    mlcs.plotsurfaces([surf_high, surf_low, surf_high_model, surf_low_model], shape=(2,2),
                      titles=['high', 'low', 'high (hierarchical model)', 'low (model)'], all_points=points,
                      save_as=save_as)
