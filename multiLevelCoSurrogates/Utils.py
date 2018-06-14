


from collections import namedtuple
from multiprocessing import cpu_count, Pool
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from warnings import warn


# ------------------------------------------------------------------------------

ValueRange = namedtuple('ValueRange', ['min', 'max'])

def determinerange(values):
    """Determine the range of values in each dimension"""
    return ValueRange(np.min(values, axis=0), np.max(values, axis=0))


def linearscaletransform(values, *, range_in=None, range_out=ValueRange(0, 1)):
    """Perform a scale transformation of `values`: [range_in] --> [range_out]"""

    if range_in is None:
        range_in = determinerange(values)
    elif not isinstance(range_in, ValueRange):
        range_in = ValueRange(*range_in)

    if not isinstance(range_out, ValueRange):
        range_out = ValueRange(*range_out)

    scale = range_out.max - range_out.min
    scaled_values = (values - range_in.min) / (range_in.max - range_in.min)
    scaled_values = (scaled_values * scale) + range_out[0]

    return scaled_values


# ------------------------------------------------------------------------------

Surface = namedtuple('Surface', ['X', 'Y', 'Z'])

def diffsurface(a, b):
    return Surface(a.X, a.Y, a.Z - b.Z)


def createsurface(func, l_bound=None, u_bound=None, step=None):
    if isinstance(func, Surface):
        return func

    l_bound = [-5, -5] if l_bound is None else l_bound
    u_bound = [5, 5] if u_bound is None else u_bound
    step = [0.1, 0.1] if step is None else step

    X = np.arange(l_bound[0], u_bound[0], step[0])
    Y = np.arange(l_bound[1], u_bound[1], step[1])
    X, Y = np.meshgrid(X, Y)
    shape = X.shape
    Z = np.array([func([[x, y]]) for x, y in zip(X.flatten(), Y.flatten())]).reshape(shape)

    return Surface(X, Y, Z)


def plotsurface(func, title=''):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = createsurface(func)
    surface = plotsurfaceonaxis(ax, surf, title)

    # Add a color bar which maps values to colors.
    fig.colorbar(surface, shrink=0.5, aspect=5)

    plt.show()


def plotsurfaces(funcs, titles=None, shape=None, figratio=(2,3)):
    if titles is None:
        titles = ['']*len(funcs)

    if shape is not None:
        if np.product(shape) != len(funcs):
            raise ValueError(f"Given shape 'np.product({shape})={np.product(shape)}'"
                             f" does not match number of functions '{len(funcs)}' given.")
    else:
        shape = (1, len(funcs))


    fig, axes = plt.subplots(*shape, figsize=(shape[0]*figratio[0], shape[1]*figratio[1]), subplot_kw={'projection': '3d'})

    with Pool(cpu_count()) as p:
        surfaces = p.map(createsurface, funcs)

    for ax, surface, title in zip(axes.flatten(), surfaces, titles):
        plotsurfaceonaxis(ax, surface, title)
        # fig.colorbar(surface, shrink=0.5, aspect=5)

    plt.show()


def plotsurfaceonaxis(ax, surf, title):

    surface = ax.plot_surface(surf.X, surf.Y, surf.Z, cmap=cm.plasma,
                              linewidth=0, antialiased=True)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(title)
    return surface
