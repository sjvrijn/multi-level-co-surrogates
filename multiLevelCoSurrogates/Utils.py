from collections import namedtuple
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pathlib import Path
from pyDOE import lhs

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


# ------------------------------------------------------------------------------

ScatterPoints = namedtuple('ScatterPoints', ['x_y', 'z', 'style'])
ValueRange = namedtuple('ValueRange', ['min', 'max'])

Surface = namedtuple('Surface', ['X', 'Y', 'Z'])

def add_surface(a, b):
    return Surface(a.X, a.Y, a.Z + b.Z)

def subtract_surface(a, b):
    return Surface(a.X, a.Y, a.Z - b.Z)

Surface.__add__ = add_surface
Surface.__sub__ = subtract_surface


# ------------------------------------------------------------------------------

def guaranteeFolderExists(path_name):
    """ Make sure the given path exists after this call """
    path = Path(path_name)
    path.expanduser()
    path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------------------

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
    distm = sp.spatial.distance.cdist(xdata.T, xdata.T, 'euclidean')
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


def create_subsample_set(ndim, size_per_fidelity, desired_range=None):
    size_per_fidelity = iter(sorted(size_per_fidelity, key=lambda x: x[1], reverse=True))
    range_lhs = ValueRange(0, 1)

    fid, size = next(size_per_fidelity)
    sample = lhs(n=ndim, samples=size)
    if desired_range is not None:
        sample = rescale(sample, range_in=range_lhs, range_out=desired_range)
    samples = {fid: sample}
    for fid, size in size_per_fidelity:
        sample = select_subsample(sample.T, num=size).T
        samples[fid] = sample

    return samples


def create_random_sample_set(ndim, size_per_fidelity, desired_range=None):
    size_per_fidelity = iter(sorted(size_per_fidelity, key=lambda x: x[1], reverse=True))
    range_lhs = ValueRange(0, 1)

    fid, size = next(size_per_fidelity)
    sample = lhs(n=ndim, samples=size)
    if desired_range is not None:
        sample = rescale(sample, range_in=range_lhs, range_out=desired_range)
    samples = {fid: sample}
    for fid, size in size_per_fidelity:
        sample = sample[np.random.choice(sample.shape[0], size=size, replace=False)]
        samples[fid] = sample

    return samples


def sample_by_function(func, n_samples, ndim, range_in, range_out, *,
                       min_p=0.01, max_p=0.99, minimize=True,
                       oversampling_factor=2.5):
    """Create a sample of points, such that they are more likely to have a
    better fitness value according to the given function.

    First a larger uniform sample (default oversampling_factor=2.5) is
    created and evaluated on the given function. These values are then
    scaled from `range_out` to [min_p, max_p], set as [0.01, 0.99] by
    default. Each of these values is then filtered using a new uniformly
    random value between [0,1] to determine whether the sampled point is
    kept or not.

    If enough valid samples remain, return them, otherwise extends the set
    of valid samples by repeating the above process until enough valid
    samples have been generated.

    :param func:                Function to sample for. Should accept
                                numpy arrays of shape `(n_samples, ndim)`.
    :param n_samples:           Desired number of samples as output.
    :param ndim:                Dimensionality of the domain.
    :param range_in:            Domain of the function to sample for.
    :param range_out:           Range of the function to sample for.
    :param min_p:               Minimum acceptance probability for a sample
                                at the worst possible function value.
                                Default: `0.01`
    :param max_p:               Maximum acceptance probability for a sample
                                at the function's optimum.
                                Default: `0.99`
    :param minimize:            Whether the function should be minimized.
                                Default: `True`.
    :param oversampling_factor: How many more samples to create per
                                iteration, reduces number of loops.
                                Default: `2.5`

    :returns:                   `(n_samples, ndim)`-array of selected samples.
    """
    range_in = ValueRange(*range_in)
    range_out = ValueRange(*range_out)

    min_p = max(min_p, 0)
    max_p = min(max_p, 1)
    range_p = ValueRange(min_p, max_p)

    function_based_sample = np.array([]).reshape((0, ndim))
    sample_shape = (int(n_samples * oversampling_factor), ndim)

    while len(function_based_sample) < n_samples:
        raw_sample = np.random.uniform(high=range_in.max, low=range_in.min,
                                       size=sample_shape)
        f_values = func(raw_sample)
        f_probabilities = rescale(f_values, range_in=range_out,
                                  range_out=range_p)

        check_values = np.random.uniform(size=f_probabilities.shape)

        if minimize:
            selected_samples = f_probabilities < check_values
        else:
            selected_samples = f_probabilities > check_values

        filtered_sample = raw_sample[selected_samples]
        if ndim == 1:
            filtered_sample = filtered_sample.reshape(-1, 1)

        function_based_sample = np.vstack((function_based_sample, filtered_sample))

    return function_based_sample[:n_samples]


# ------------------------------------------------------------------------------

def determinerange(values):
    """Determine the range of values in each dimension"""
    return ValueRange(np.min(values, axis=0), np.max(values, axis=0))


def rescale(values, *, range_in=None, range_out=ValueRange(0, 1), scale_only=False):
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
        scaled_values = (scaled_values * scale_out) + range_out.min

    return scaled_values


# ------------------------------------------------------------------------------

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

    l_bound = np.array([-5, -5]) if l_bound is None else l_bound
    u_bound = np.array([5, 5]) if u_bound is None else u_bound
    step = [0.2, 0.2] if step is None else step

    X, Y = create_wide_meshgrid(l_bound, step, u_bound)

    X_Y = np.array([[[x,y]] for x, y in zip(X.flatten(), Y.flatten())])
    Z = np.array([func(x_y) for x_y in X_Y]).reshape(X.shape)

    return Surface(X, Y, Z)


def createsurfaces(funcs, *args, **kwargs):
    """Create Surface objects for each function in the list `funcs` using the default parameters"""
    return [createsurface(func, *args, **kwargs) for func in funcs]


# ------------------------------------------------------------------------------

def plotsurfaces(surfaces, *, all_points=None, titles=None, shape=None, figratio=None, save_as=None, as_3d=True, show=True, **_):
    """Plot a set of surfaces as subfigures in a single figure"""
    if titles is None:
        titles = ['']*len(surfaces)
    elif len(titles) < len(surfaces):
        titles.extend([''] * (len(surfaces) - len(titles)))

    if shape is not None:
        if np.product(shape) != len(surfaces):
            raise ValueError(f"Given shape 'np.product({shape})={np.product(shape)}'"
                             f" does not match number of functions '{len(surfaces)}' given.")
    else:
        shape = (1, len(surfaces))

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
    plt.clf()


def plotsurfaceonaxis(ax, surf, title, point_sets=None, plot_type='wireframe', contour=False):
    """Plot a Surface as 3D surface on a given matplotlib Axis"""

    offset = np.min(surf.Z)
    rows, cols = surf.X.shape

    if plot_type == 'wireframe':
        colors = cm.viridis_r(rescale(surf.Z))
        surface = ax.plot_surface(surf.X, surf.Y, surf.Z, rcount=15, ccount=15,
                                  facecolors=colors, shade=False)
        surface.set_facecolor((0,0,0,0))
    elif plot_type == 'surface':
        surface = ax.plot_surface(surf.X, surf.Y, surf.Z, cmap='viridis_r', rcount=rows, ccount=cols,
                                  linewidth=0, antialiased=True)

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

    surface = ax.pcolormesh(surf.X, surf.Y, surf.Z, cmap=cm.viridis)
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
