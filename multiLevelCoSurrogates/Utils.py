


from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


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


def plotsurfaces(funcs, titles=None, shape=None):
    if titles is None:
        titles = ['']*len(funcs)

    if shape is not None:
        if np.product(shape) != len(funcs):
            raise ValueError(f"Given shape 'np.product({shape})={np.product(shape)}'"
                             f" does not match number of functions '{len(funcs)}' given.")
    else:
        shape = (1, len(funcs))


    fig, axes = plt.subplots(*shape, figsize=plt.figaspect(shape[0]/shape[1]), subplot_kw={'projection': '3d'})

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