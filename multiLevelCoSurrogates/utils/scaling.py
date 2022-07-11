from collections import namedtuple
from typing import Sequence, List

import numpy as np

from mf2 import MultiFidelityFunction

ValueRange = namedtuple('ValueRange', ['min', 'max'])


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


def scale_to_function(func: MultiFidelityFunction,
                      xx: Sequence[Sequence],
                      range_in=ValueRange(0, 1)) -> List[Sequence]:
    """Scale the input data `xx` from `range_in` to the bounds of the given function.
    :param range_in: defined range from which input values were drawn. Default: (0,1)
    """
    range_out = (np.array(func.l_bound), np.array(func.u_bound))
    return [rescale(x, range_in=range_in, range_out=range_out) for x in xx]