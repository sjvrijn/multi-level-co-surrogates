from collections import namedtuple
from operator import mul
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def select_subsample(xdata, num):
    """
    uniform selection of sub samples from a larger data set (only for input).
    use it to create uniform sample
       inputs:
          xdata  : inputs data set. each row is a dimension
            num  : final (desired) number of samples

    Based on : Rennen, G. (2008). "SUBSET SELECTION FROM LARGE DATASETS
              FOR KRIGING MODELING.

    Acknowledgement --- Code kindly provided by:
        Koushyar Komeilizadeh, M.Sc.
        Research Assistant
        Associate Professorship of Computational Mechanics
        Faculty of Civil, Geo and Environmental Engineering
        Technical University of Munich
    """
    # if initial sample number is large convert to float 32, otherwise distance
    # matrix might not be calculated, # float 32 precision is 10^-6, float 16 is 10^-3
    if xdata.size > 10000:
        xdata = np.float32(xdata)
    distm = scipy.spatial.distance.cdist(xdata.T, xdata.T, 'euclidean')
    include = np.where(distm == np.max(distm))[0]
    si = np.arange(xdata.shape[1])
    remain = np.delete(si, include)
    for j in range(num - 2):
        minr = np.zeros([np.size(remain)])
        minrind = np.zeros([np.size(remain)])
        for i, ind in enumerate(remain):
            minr[i] = np.min((distm[ind, include.astype(int)]))
            minrind[i] = ind
        minminrind = np.argmax(minr)
        include = np.append(include, minrind[minminrind])
        remain = np.delete(si, include)
    sub_x = xdata[:, list(map(int, include[:num]))]

    return sub_x


# ------------------------------------------------------------------------------

ValueRange = namedtuple('ValueRange', ['min', 'max'])

def determinerange(values):
    """Determine the range of values in each dimension"""
    return ValueRange(np.min(values, axis=0), np.max(values, axis=0))


def linearscaletransform(values, *, range_in=None, range_out=ValueRange(0, 1), scale_only=False):
    """Perform a scale transformation of `values`: [range_in] --> [range_out]"""

    if range_in is None:
        range_in = determinerange(values)
    elif not isinstance(range_in, ValueRange):
        range_in = ValueRange(*range_in)

    if not isinstance(range_out, ValueRange):
        range_out = ValueRange(*range_out)

    scale_out = range_out.max - range_out.min
    scale_in = range_in.max - range_in.min

    if scale_only:
        scaled_values = (values / scale_in) * scale_out
    else:
        scaled_values = (values - range_in.min) / scale_in
        scaled_values = (scaled_values * scale_out) + range_out[0]

    return scaled_values


# ------------------------------------------------------------------------------

Surface = namedtuple('Surface', ['X', 'Y', 'Z'])

def diffsurface(a, b):
    """Calculate the difference/response surface between surfaces a and b"""
    return Surface(a.X, a.Y, a.Z - b.Z)


def calc_numsteps(low, high, step, endpoint=True):
    """Calculate the number of 'step' steps between 'low' and 'high'"""
    num_steps = (high - low) / step
    if endpoint:
        num_steps += 1
    return num_steps


def create_wide_meshgrid(l_bound, step, u_bound):
    """Create a meshgrid that extends 1 step in each direction beyond the original bounds"""
    num_steps_x = calc_numsteps(l_bound[0], u_bound[0], step[0]) + 2
    num_steps_y = calc_numsteps(l_bound[1], u_bound[1], step[1]) + 2
    X = np.linspace(l_bound[0] - step[0], u_bound[0] + step[0], num_steps_x)
    Y = np.linspace(l_bound[1] - step[1], u_bound[1] + step[1], num_steps_y)
    X, Y = np.meshgrid(X, Y)
    return X, Y


def createsurface(func, l_bound=None, u_bound=None, step=None):
    """Create a Surface(X, Y, Z) by evaluating `func` on a (wide) grid ranging from l_bound to u_bound"""
    if isinstance(func, Surface):
        return func

    l_bound = [-5, -5] if l_bound is None else l_bound
    u_bound = [5, 5] if u_bound is None else u_bound
    step = [0.2, 0.2] if step is None else step

    X, Y = create_wide_meshgrid(l_bound, step, u_bound)

    X_Y = np.array([[[x,y]] for x, y in zip(X.flatten(), Y.flatten())])
    Z = np.array([func(x_y) for x_y in X_Y]).reshape(X.shape)

    return Surface(X, Y, Z)


def createsurfaces(funcs):
    """Create Surface objects for each function in the list `funcs` using the default parameters"""
    return [createsurface(func) for func in funcs]


# ------------------------------------------------------------------------------

def plotsurfaces(surfaces, *, all_points=None, titles=None, shape=None, figratio=None, save_as=None, as_3d=True, show=True, **_):
    """Plot a set of surfaces as subfigures in a single figure"""
    if titles is None:
        titles = ['']*len(surfaces)

    if shape is not None:
        if np.product(shape) != len(surfaces):
            raise ValueError(f"Given shape 'np.product({shape})={np.product(shape)}'"
                             f" does not match number of functions '{len(surfaces)}' given.")
    else:
        shape = (1, len(surfaces))

    if as_3d:
        kwargs = {'projection': '3d'}
        plot_func = plotsurfaceonaxis
        figratio = (3,4.5) if figratio is None else figratio
    else:
        kwargs = dict()
        plot_func = plotcmaponaxis
        figratio = (2.25,4.5) if figratio is None else figratio

    if all_points is None:
        all_points = [None] * len(surfaces)

    fig, axes = plt.subplots(*shape, figsize=(shape[0]*figratio[0], shape[1]*figratio[1]), subplot_kw=kwargs)
    for ax, surface, title, points in zip(axes.flatten(), surfaces, titles, all_points):
        plot = plot_func(ax, surface, title, points)
        if not as_3d:
            fig.colorbar(plot, ax=ax)

    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as)
    if show:
        plt.show()
    plt.clf()


def plotsurfaceonaxis(ax, surf, title, point_sets=None):
    """Plot a Surface as 3D surface on a given matplotlib Axis"""

    surface = ax.plot_surface(surf.X, surf.Y, surf.Z, cmap=cm.viridis,
                              linewidth=0, antialiased=True)
    if point_sets:
        for x_y, z, style in point_sets:
            ax.scatter(x_y[:, 0], x_y[:, 1], z[1], **style)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(title)
    return surface


def plotcmaponaxis(ax, surf, title, point_sets=None):
    """Plot a Surface as 2D heatmap on a given matplotlib Axis"""

    surface = ax.pcolor(surf.X, surf.Y, surf.Z, cmap=cm.viridis)
    if point_sets:
        for x_y, z, style in point_sets:
            ax.scatter(x_y[:, 0], x_y[:, 1], **style)
    ax.set_title(title)
    return surface
