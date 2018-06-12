


from collections import namedtuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from warnings import warn



def get_min_and_scale(values):
    """Determine the minimum value and scale (max - min) in each dimension"""
    warn(DeprecationWarning('will be deprecated'))
    min_vals = np.min(values, axis=0)
    max_vals = np.max(values, axis=0)

    scale = max_vals - min_vals

    return min_vals, scale


def normalize(values, *, min_vals=None, scale=None, target_range=(0.25, 0.75)):
    """Normalize the given values to target range (default: [0.25, 0.75])"""
    warn(DeprecationWarning('will be deprecated'))

    if (min_vals is not None) is not (scale is not None):  # x-or: if just one of them is None
        raise Exception("If specified, both 'min_vals' and 'scale' must be given.")
    elif min_vals is None and scale is None:
        min_vals, scale = get_min_and_scale(values)

    if any(scale == 0):  # Hardcoded error prevention TODO: find better solution?!?
        scale[scale == 0] = 1
        # raise ValueError(f'Scale cannot be 0: {scale}')

    t_min, t_max = target_range

    normalized_values = (values - min_vals) / scale                    # Normalize to [0,1]
    normalized_values = (normalized_values * (t_max - t_min)) + t_min  # Scale to [t_min, t_max]

    return normalized_values


def denormalize(values, min_vals, scale, *, target_range=(0.25, 0.75)):
    """Denormalize the given normalized values, default assumed normalization target is [0.25, 0.75]"""
    warn(DeprecationWarning('will be deprecated'))

    t_min, t_max = target_range

    denormalized_values = (values - t_min) / (t_max - t_min)         # Reverse [t_min, t_max] to [0,1] scale
    denormalized_values = (denormalized_values * scale) + min_vals   # Denormalize to original range

    return denormalized_values


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

def createsurface(l_bound, u_bound, step, func):
    X = np.arange(l_bound[0], u_bound[0], step[0])
    Y = np.arange(l_bound[1], u_bound[1], step[1])
    X, Y = np.meshgrid(X, Y)
    shape = X.shape
    Z = np.array([func([[x, y]]) for x, y in zip(X.flatten(), Y.flatten())]).reshape(shape)

    return Surface(X, Y, Z)


def plotsurface(func, title=''):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surface = plotfunctiononaxis(ax, func, title)

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

    for ax, func, title in zip(axes.flatten(), funcs, titles):
        surface = plotfunctiononaxis(ax, func, title)

        # Add a color bar which maps values to colors.
        # fig.colorbar(surface, shrink=0.5, aspect=5)

    plt.show()



def plotfunctiononaxis(ax, func, title, *, l_bound=None, u_bound=None, step=None):

    l_bound = [-5, -5] if l_bound is None else l_bound
    u_bound = [5, 5] if u_bound is None else u_bound
    step = [0.1, 0.1] if step is None else step

    surf = createsurface(l_bound=l_bound, u_bound=u_bound, step=step, func=func)

    return plotsurfaceonaxis(ax, surf, title)


def plotsurfaceonaxis(ax, surf, title):

    # Plot the surface.
    surface = ax.plot_surface(surf.X, surf.Y, surf.Z, cmap=cm.plasma,
                              linewidth=0, antialiased=True)
    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(title)
    return surface