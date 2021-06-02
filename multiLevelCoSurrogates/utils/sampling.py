from collections import namedtuple
from pyDOE import lhs
from scipy.spatial import distance
from typing import Tuple
import numpy as np
import scipy as sp


BiFidelityDoE = namedtuple("BiFidelityDoE", "high low")


def low_lhs_sample(ndim, nlow):
    if ndim == 1:
        return np.linspace(0,1,nlow).reshape(-1,1)
    elif ndim > 1:
        return lhs(ndim, nlow)


def bi_fidelity_doe(ndim: int, num_high: int, num_low: int) -> BiFidelityDoE:
    """Create a Design of Experiments (DoE) for two fidelities in `ndim`
    dimensions. The high-fidelity samples are guaranteed to be a subset
    of the low-fidelity samples.

    :returns high-fidelity samples, low-fidelity samples
    """
    high_x = low_lhs_sample(ndim, num_high)
    low_x = low_lhs_sample(ndim, num_low)

    dists = distance.cdist(high_x, low_x)

    #TODO: this is the naive method, potentially speed up?
    highs_to_match = set(range(num_high))
    while highs_to_match:
        min_dist = np.min(dists)
        high_idx, low_idx = np.argwhere(dists == min_dist)[0]

        low_x[low_idx] = high_x[high_idx]
        # make sure just selected samples are not re-selectable
        dists[high_idx,:] = np.inf
        dists[:,low_idx] = np.inf
        highs_to_match.remove(high_idx)
    return BiFidelityDoE(high_x, low_x)


def split_bi_fidelity_doe(DoE: BiFidelityDoE, num_high: int, num_low: int) -> Tuple[BiFidelityDoE, BiFidelityDoE]:
    """Given an existing bi-fidelity Design of Experiments (DoE) `high, low`,
    creates a subselection of given size `num_high, num_low` based on uniform
    selection. The subselection maintains the property that all high-fidelity
    samples are a subset of the low-fidelity samples.

    Raises a `ValueError` if invalid `num_high` or `num_low` are given.
    """
    high, low = DoE
    if not 1 < num_high < len(high):
        raise ValueError(f"'num_high' must be in the range [2, len(DoE.high) (={len(DoE.high)})], but is {num_high}")
    elif num_low > len(low):
        raise ValueError(f"'num_low' cannot be greater than len(DoE.low) (={len(DoE.low)}), but is {num_low}")
    elif num_low <= num_high:
        raise ValueError(f"'num_low' must be greater than 'num_high', but {num_low} <= {num_high}")

    indices = np.random.permutation(len(high))
    sub_high, leave_out_high = high[indices[:num_high]], high[indices[num_high:]]

    if num_low == len(low):
        sub_low = low
        leave_out_low = []
    else:
        # remove all sub_high from low
        filtered_low = np.array([x for x in low if x not in sub_high])
        # randomly select (num_low - num_high) remaining
        indices = np.random.permutation(len(filtered_low))
        num_low_left = num_low - num_high
        extra_low, leave_out_low = filtered_low[indices[:num_low_left]], \
                                   filtered_low[indices[num_low_left:]]
        # concatenate sub_high with selected sub_low
        sub_low = np.concatenate([sub_high, extra_low], axis=0)

    selected = BiFidelityDoE(sub_high, sub_low)
    left_out = BiFidelityDoE(leave_out_high, leave_out_low)
    return selected, left_out


def select_subsample(xdata, num):
    """Uniform selection of sub samples from a larger data set (only for input).
    Use it to create a uniform sample
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