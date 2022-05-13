from collections import namedtuple
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import matplotlib.pyplot as plt
import numpy as np

from .scaling import rescale


ScatterPoints = namedtuple('ScatterPoints', ['x_y', 'z', 'style'])


Surface = namedtuple('Surface', ['X', 'Y', 'Z'])

def add_surface(a, b):
    return Surface(a.X, a.Y, a.Z + b.Z)

def subtract_surface(a, b):
    return Surface(a.X, a.Y, a.Z - b.Z)

Surface.__add__ = add_surface
Surface.__sub__ = subtract_surface


def plotsurfaces(surfaces, *, all_points=None, titles=None, shape=None,
                 figratio=None, save_as=None, as_3d=True, show=True, **_):
    """Plot a set of surfaces as subfigures in a single figure"""
    if titles is None:
        titles = ['']*len(surfaces)
    elif len(titles) < len(surfaces):
        titles.extend([''] * (len(surfaces) - len(titles)))

    if shape is None:
        shape = (1, len(surfaces))
    elif np.product(shape) != len(surfaces):
        raise ValueError(f"Given shape 'np.product({shape})={np.product(shape)}'"
                         f" does not match number of functions '{len(surfaces)}' given.")

    figratio = (5,4.5) if figratio is None else figratio
    if as_3d:
        kwargs = {'projection': '3d'}
        plot_func = plotsurfaceonaxis
    else:
        kwargs = dict()
        plot_func = plotcmaponaxis
        # figratio = (2.25,4.5) if figratio is None else figratio

    if all_points is None:
        all_points = [None] * len(surfaces)

    nrows, ncols = shape
    single_width, single_height = figratio
    width = single_width*ncols
    height = single_height*nrows

    fig, axes = plt.subplots(*shape, figsize=(width, height), subplot_kw=kwargs)
    try:
        axes = axes.flatten()
    except AttributeError:
        axes = [axes]

    for ax, surface, title, points in zip(axes, surfaces, titles, all_points):
        plot = plot_func(ax, surface, title, points)
        if not as_3d:
            fig.colorbar(plot, ax=ax)

    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.close()


def plotsurfaceonaxis(ax, surf, title, point_sets=None, plot_type='wireframe', contour=False):
    """Plot a Surface as 3D surface on a given matplotlib Axis"""

    offset = np.min(surf.Z)
    rows, cols = surf.X.shape

    if plot_type == 'wireframe':
        colors = cm.viridis_r(rescale(surf.Z))
        surface = ax.plot_surface(surf.X, surf.Y, surf.Z,
                                  rstride=1, cstride=1,
                                  # rcount=15, ccount=15,
                                  # shade=False,
                                  cmap=cm.viridis_r,
                                  # facecolors=colors,
                                  )
        surface.set_facecolor((0,0,0,0))
    elif plot_type == 'surface':
        surface = ax.plot_surface(surf.X, surf.Y, surf.Z, cmap='viridis_r', rcount=rows, ccount=cols,
                                  linewidth=0, antialiased=True)
    else:
        raise ValueError(f"Unrecognised plot_type: '{plot_type}'")

    if contour:
        ax.contour(surf.X, surf.Y, surf.Z, zdir='z', levels=33,
                   offset=offset, cmap='viridis_r')
    if point_sets:
        for x_y, z, style in point_sets:
            ax.scatter(x_y[:, 0], x_y[:, 1], z, **style)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    return surface


def plotcmaponaxis(ax, surf, title, point_sets=None):
    """Plot a Surface as 2D heatmap on a given matplotlib Axis"""

    surface = ax.pcolormesh(surf.X, surf.Y, surf.Z, cmap=cm.viridis_r)
    if point_sets:
        for x_y, z, style in point_sets:
            ax.scatter(x_y[:, 0], x_y[:, 1], **style)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    return surface


def gpplot(x, func, return_std=False):
    idx = 1 if return_std else 0
    return func(x, return_std=return_std)[idx]


def calc_numsteps(low, high, step, endpoint=True):
    """Calculate the number of 'step' steps between 'low' and 'high'"""
    num_steps = (high - low) // step
    if endpoint:
        num_steps += 1
    return int(num_steps)


def create_meshgrid(l_bound, step, u_bound):
    """Create a meshgrid within the given bounds"""
    num_steps_x = calc_numsteps(l_bound[0], u_bound[0], step[0])
    num_steps_y = calc_numsteps(l_bound[1], u_bound[1], step[1])
    X = np.linspace(l_bound[0], u_bound[0], num_steps_x)
    Y = np.linspace(l_bound[1], u_bound[1], num_steps_y)
    X, Y = np.meshgrid(X, Y)
    return X, Y


def create_wide_meshgrid(l_bound, step, u_bound):
    """Create a meshgrid that extends 1 step in each direction beyond the
    original bounds
    """
    num_steps_x = calc_numsteps(l_bound[0], u_bound[0], step[0]) + 2
    num_steps_y = calc_numsteps(l_bound[1], u_bound[1], step[1]) + 2
    X = np.linspace(l_bound[0] - step[0], u_bound[0] + step[0], num_steps_x)
    Y = np.linspace(l_bound[1] - step[1], u_bound[1] + step[1], num_steps_y)
    X, Y = np.meshgrid(X, Y)
    return X, Y


def createsurface(func, l_bound=None, u_bound=None, step=None, wide=True):
    """Create a Surface(X, Y, Z) by evaluating `func` on a (wide) grid ranging
    from l_bound to u_bound
    """
    if isinstance(func, Surface):
        return func

    l_bound = np.array([-5, -5]) if l_bound is None else l_bound
    u_bound = np.array([5, 5]) if u_bound is None else u_bound
    step = [0.2, 0.2] if step is None else step

    if wide:
        X, Y = create_wide_meshgrid(l_bound, step, u_bound)
    else:
        X, Y = create_meshgrid(l_bound, step, u_bound)

    X_Y = np.array([[[x,y]] for x, y in zip(X.flatten(), Y.flatten())])
    Z = np.array([func(x_y) for x_y in X_Y]).reshape(X.shape)

    return Surface(X, Y, Z)


def createsurfaces(funcs, *args, **kwargs):
    """Create Surface objects for each function in the list `funcs` using the
    default parameters
    """
    return [createsurface(func, *args, **kwargs) for func in funcs]
