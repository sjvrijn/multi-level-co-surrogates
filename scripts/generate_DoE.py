#!/usr/bin/env python3

"""Standalone script to generate multi-fidelity DoEs"""

from argparse import ArgumentParser
from collections import namedtuple
from csv import writer
from pathlib import Path

import numpy as np
from pyDOE import lhs
from scipy.spatial import distance


BiFidelityDoE = namedtuple("BiFidelityDoE", "high low")
ValueRange = namedtuple('ValueRange', ['min', 'max'])

BASE_SEED = 2016_05_01  # PhD start date as unbiased random seed


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


def low_lhs_sample(ndim: int, nlow: int):
    if ndim == 1:
        return np.linspace(0,1,nlow).reshape(-1,1)
    elif ndim > 1:
        return lhs(ndim, nlow)

def determinerange(values):
    """Determine the range of values in each dimension"""
    r = ValueRange(np.min(values, axis=0), np.max(values, axis=0))
    if np.any(r.max - r.min < 1e-8):
        r = ValueRange(r.min - 1e8, r.max + 1e8)
    return r


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


def save_DOE(doe, filename):
    with open(filename, 'w', newline='') as csvfile:
        doe_writer = writer(csvfile)
        doe_writer.writerow(['x{}'.format(i) for i in range(len(doe[0]))])
        for row in doe:
            doe_writer.writerow(row)


def save_bi_fid_DOE(doe, base_filename):
    if not base_filename.suffix:
        base_filename = base_filename.with_suffix('.csv')

    directory, filename = base_filename.parent, base_filename.name
    directory.mkdir(exist_ok=True, parents=True)

    save_DOE(doe.high, directory / ('high_' + filename))
    save_DOE(doe.low, directory / ('low_' + filename))


def generate_DOE(
    ndim: int,
    num_high: int,
    num_low: int,
    range_out=ValueRange(-1, 1),
    seed_offset=0,
):
    
    np.random.seed(BASE_SEED + seed_offset)
    raw_doe = bi_fidelity_doe(ndim, num_high, num_low)
    scaled_doe = BiFidelityDoE(
        rescale(raw_doe.high, range_in=ValueRange(0, 1), range_out=range_out),
        rescale(raw_doe.low, range_in=ValueRange(0, 1), range_out=range_out),
    )

    return scaled_doe


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--ndim', default=4, type=int)
    parser.add_argument('--nhigh', default=50, type=int)
    parser.add_argument('--nlow', default=125, type=int)
    parser.add_argument('--min', default=None, type=float)
    parser.add_argument('--max', default=0.25, type=float)
    parser.add_argument('-o', '--output', default=None, type=str)
    parser.add_argument('-s', '--seed', default=0, type=int)
    args = parser.parse_args()

    if args.min is None:
        args.min = -float(args.max)
    if args.output is None:
        args.output = f'{args.ndim}D_DoE.csv'

    range_out = ValueRange(args.min, args.max)

    doe = generate_DOE(args.ndim, args.nhigh, args.nlow, range_out=range_out, seed_offset=args.seed)
    save_bi_fid_DOE(doe, Path(args.output))
