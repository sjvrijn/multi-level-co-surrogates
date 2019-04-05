from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyDOE import lhs
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import sys
import warnings

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from itertools import product
from more_itertools import chunked
from functools import partial
import multifidelityfunctions as mff
import multiLevelCoSurrogates as mlcs
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.ensemble import RandomForestRegressor

np.random.seed(20160501)  # Setting seed for reproducibility
OD = mff.oneDimensional

from IPython.core.display import clear_output


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
        archive.addcandidates(low_x, func.low(low_x), fidelity='low')
        archive.addcandidates(high_x, func.high(high_x), fidelity='high')

        mfbo = mlcs.MultiFidelityBO(func, archive, output_range=(-10, 16))
        mse_tracking[num_high, num_low, rep] = mfbo.getMSE()

    clear_output()
    print(f'{len(cases)}/{len(cases)}')
    return mse_tracking


def plot_high_vs_low_num_samples(data, name, vmin=.5, vmax=100, save_as=None):
    norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    fig, ax = plt.subplots(figsize=(9,3.5))

    ax.set_aspect(1.)
    data = np.nanmedian(data, axis=2)

    plt.title('Median MSE for high (hierarchical) model')
    img = ax.imshow(data[:,:,0], cmap='viridis_r', norm=norm)

    divider = make_axes_locatable(ax)
    axx = divider.append_axes("bottom", size=.2, pad=0.05, sharex=ax)
    axy = divider.append_axes("left", size=.2, pad=0.05, sharey=ax)

    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    axy.xaxis.set_tick_params(labelbottom=False)
    axx.yaxis.set_tick_params(labelleft=False)

    img = axy.imshow(np.nanmean(data[:,:,1], axis=1).reshape(-1,1), cmap='viridis_r', norm=norm)
    img = axx.imshow(np.nanmean(data[:,:,2], axis=0).reshape(1,-1), cmap='viridis_r', norm=norm)

    fig.colorbar(img, ax=ax, orientation='vertical')
    axy.set_ylabel('#High-fid samples')
    axx.set_xlabel('#Low-fid samples')

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)  # f'{plot_dir}{name}.pdf'
    plt.show()


def plot_high_vs_low_num_samples_diff(data, name, vmin=.5, vmax=100, save_as=None):

    to_plot = np.nanmedian(data[:,:,:,1] - data[:,:,:,0], axis=2)
    max_diff = 2*min(abs(np.nanmin(to_plot)), np.nanmax(to_plot))
    norm = colors.Normalize(vmin=-max_diff, vmax=max_diff, clip=True)

    fig, ax = plt.subplots(figsize=(9,3.5))
    img = ax.imshow(to_plot, cmap='RdYlGn', norm=norm)
    fig.colorbar(img, ax=ax, orientation='vertical')
    ax.set_ylabel('#High-fid samples')
    ax.set_xlabel('#Low-fid samples')

    plt.title('Median of paired (high (hierarchical) - high (direct)) MSE')
    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)  # f'{plot_dir}{name}_diff.pdf'
    plt.show()


def plot_inter_method_diff(data_A, data_B, name, save_as=None):
    fig, ax = plt.subplots(figsize=(9,3.5))

    plt.title(f'high (hierarchical) MSE: {name}')
    to_plot = np.nanmedian(data_A[:,:,:,0] - data_B[:,:,:,0], axis=2)

    max_diff = .05*min(abs(np.nanmin(to_plot)), np.nanmax(to_plot))
    norm = colors.Normalize(vmin=-max_diff, vmax=max_diff, clip=True)

    img = ax.imshow(to_plot, cmap='RdYlGn', norm=norm)

    fig.colorbar(img, ax=ax, orientation='vertical')
    ax.set_ylabel('#High-fid samples')
    ax.set_xlabel('#Low-fid samples')

    plt.tight_layout()
    if save_as:
        plt.savefig(save_as)  # f'{plot_dir}{name}.pdf'
    plt.show()


# defining some point styles
red_dot = {'marker': '.', 'color': 'red'}
blue_circle = {'marker': 'o', 'facecolors': 'none', 'color': 'blue'}

@mff.row_vectorize
def td_inv_high(xx):
    x1, x2 = xx
    return -(OD.high(x1) + OD.high(x2))

@mff.row_vectorize
def td_inv_low(xx):
    x1, x2 = xx
    return -(OD.low(x1) + OD.low(x2))


TD_inv = mff.MultiFidelityFunction(
    u_bound=np.array(OD.u_bound*2), l_bound=np.array(OD.l_bound*2),
    functions=[td_inv_high, td_inv_low],
    fidelity_names=['high', 'low'],
)


def create_models_and_compare(func, low, high, save_as=None):
    archive = mlcs.CandidateArchive(ndim=2, fidelities=['high', 'low', 'high-low'])
    archive.addcandidates(low, func.low(low), fidelity='low')
    archive.addcandidates(high, func.high(high), fidelity='high')

    mfbo = mlcs.MultiFidelityBO(func, archive, output_range=(-16, 10), schema=[1,1])

    surf_high_model = mlcs.createsurface(mfbo.models['high'].predict, u_bound=u_bound, l_bound=l_bound, step=steps)
    surf_low_model = mlcs.createsurface(mfbo.models['low'].predict, u_bound=u_bound, l_bound=l_bound, step=steps)

    points_high = [mlcs.ScatterPoints(*archive.getcandidates(fidelity='high'), red_dot)]
    points_low = [mlcs.ScatterPoints(*archive.getcandidates(fidelity='low'), blue_circle)]

    points = [
        points_high, points_low,
        points_high, points_low,
    ]

    mlcs.plotsurfaces([surf_high, surf_low, surf_high_model, surf_low_model], shape=(2,2),
                      titles=['high', 'low', 'high (hierarchical model)', 'low (model)'], all_points=points,
                      save_as=save_as)
