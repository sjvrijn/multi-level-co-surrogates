from collections import namedtuple

import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from scipy.spatial import distance
from pyprojroot import here


BiFidelityDoE = namedtuple("BiFidelityDoE", "high low")


def low_lhs_sample(ndim, nlow):
    if ndim == 1:
        return np.linspace(0,1,nlow).reshape(-1,1)
    elif ndim > 1:
        return lhs(ndim, nlow)


def illustrated_bi_fidelity_doe(ndim, num_high, num_low, intermediate=True,
                                as_pdf=True, save_dir=None):
    """Create a Design of Experiments (DoE) for two fidelities in `ndim`
    dimensions. The high-fidelity samples are guaranteed to be a subset
    of the low-fidelity samples.

    :returns high-fidelity samples, low-fidelity samples
    """

    extension = 'pdf' if as_pdf else 'png'

    high_x = low_lhs_sample(ndim, num_high)
    low_x = low_lhs_sample(ndim, num_low)

    dists = distance.cdist(high_x, low_x)
    fig_size = (4, 4) if ndim >= 2 else (4, 2)
    plt.rcParams.update({'font.size': 16})
    plt.rc('axes', labelsize=20)

    low_style = {'s': 36}
    high_style = {'s': 288, 'marker': '+'}
    arrow_style = {
        'width': .0025,
        'head_width': .03,
        'facecolor': 'black',
        'length_includes_head': True,
    }

    #TODO: this is the naive method, potentially speed up?
    highs_to_match = set(range(num_high))
    while highs_to_match:

        min_dist = np.min(dists)
        high_idx, low_idx = np.argwhere(dists == min_dist)[0]

        if intermediate:
            xlow = low_x.T[0]
            xhigh = high_x.T[0]
            if ndim >= 2:
                ylow = low_x.T[1]
                yhigh = high_x.T[1]
            else:
                ylow = np.zeros(xlow.shape)
                yhigh = np.zeros(xhigh.shape)

            plt.figure(figsize=fig_size, constrained_layout=True)
            plt.scatter(xlow, ylow, label='low', **low_style)
            plt.scatter(xhigh, yhigh, label='high', **high_style)
            plt.arrow(
                *low_x[low_idx],
                *(high_x[high_idx] - low_x[low_idx]),
                **arrow_style,
            )
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.title(f'step {num_high-len(highs_to_match)}/{num_high}')
            if save_dir:
                plt.savefig(save_dir / f'illustrated-bi-fid-doe-{num_high-len(highs_to_match)}.{extension}')
            plt.show()
            plt.close()

        low_x[low_idx] = high_x[high_idx]
        # make sure just selected samples are not re-selectable
        dists[high_idx,:] = np.inf
        dists[:,low_idx] = np.inf
        highs_to_match.remove(high_idx)


    xlow = low_x.T[0]
    xhigh = high_x.T[0]
    if ndim >= 2:
        ylow = low_x.T[1]
        yhigh = high_x.T[1]
    else:
        ylow = np.zeros(xlow.shape)
        yhigh = np.zeros(xhigh.shape)

    plt.figure(figsize=fig_size, constrained_layout=True)
    plt.scatter(xlow, ylow, label='low', **low_style)
    plt.scatter(xhigh, yhigh, label='high', **high_style)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'step {num_high-len(highs_to_match)}/{num_high}')
    if save_dir:
        plt.savefig(save_dir / f'illustrated-bi-fid-doe-{ndim}d-{num_high}-{num_low}-{num_high-len(highs_to_match)}.{extension}')
    plt.show()

    return BiFidelityDoE(high_x, low_x)


np.random.seed(20160501)

plot_dir = here('plots') / 'illustrated-doe'
plot_dir.mkdir(exist_ok=True, parents=True)

_ = illustrated_bi_fidelity_doe(2, 10, 20, save_dir=plot_dir)
