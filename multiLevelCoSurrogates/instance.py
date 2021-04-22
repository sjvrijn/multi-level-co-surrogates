from collections import namedtuple
from dataclasses import dataclass
from itertools import product
from warnings import warn

import numpy as np


class Instance(namedtuple('Instance', 'n_high n_low rep'))

    def use_as_seed(self):
        """Fix the numpy random seed based on this Instance"""
        np.random.seed(int(f'{self.n_high:03}{self.n_low:03}{self.rep:03}'))



def set_seed_by_instance(n_high: int, n_low: int, rep: int) -> None:
    """Fix the numpy random seed based on an Instance"""
    np.random.seed(int(f'{n_high:03}{n_low:03}{rep:03}'))


@dataclass
class InstanceSpec:
    max_high:  int
    max_low:   int
    min_high:  int = 2
    min_low:   int = 3
    step_high: int = None
    step_low:  int = None
    num_reps:  int = 50

    def __init__(self, max_high, max_low,
                 *,
                 min_high=2, min_low=3,
                 num_reps=50,
                 step=1, step_high=None, step_low=None,
                 ):
        """Specification for a collection of `Instance`s"""

        self.max_high = max_high
        self.max_low = max_low
        self.min_high = min_high
        self.min_low = min_low
        self.num_reps = num_reps

        no_steps_specified = not (step or step_high or step_low)
        default_step_missing = not step and not (step_high and step_low)
        all_given = step and step_high and step_low

        if no_steps_specified:
            raise ValueError('Some step information must be defined')
        if default_step_missing:
            raise ValueError('Default `step` must be present if only one'
                             'of `step_high` and `step_low` is given')
        if all_given:
            warn('Both `step_high` and `step_low` given, default `step` is ignored')

        self.step_high = self.step_low = step
        if step_high:
            self.step_high = step_high
        if step_low:
            self.step_low = step_low


    @property
    def instances(self):
        """Generator for all valid Instances h,l,r"""
        yield from (
            Instance(h, l, r)
            for (h, l), r in product(
                self.pixels,
                range(self.num_reps),
            )
            if h < l
        )


    @property
    def pixels(self):
        """Generator for all valid 'pixels' h,l (i.e. Instance() without r)"""
        yield from (
            (h, l)
            for h, l in product(
                range(self.min_high, self.max_high + 1, self.step_high),
                range(self.min_low, self.max_low + 1, self.step_low),
            )
            if h < l
        )


    def __len__(self):
        """Number of valid instances defined by this InstanceSpec

        I.e. len(InstanceSpec.instances), but without calling and exhausting that generator
        """
        high_vals_lower_than_low_vals = self.max_high < self.max_low and self.min_high < self.min_low
        steps_are_1 = self.step_high == self.step_low == 1

        try:
            assert high_vals_lower_than_low_vals and steps_are_1
        except AssertionError:
            raise NotImplementedError("Easy len() not implemented for more complicated cases, "
                                      "use len(list(InstanceSpec.instances())) instead.")

        num_high = self.max_high - self.min_high + 1
        num_low = self.max_low - self.min_low + 1
        triangle_side_length = self.max_high - self.min_low + 1
        triangle_size = triangle_side_length * (triangle_side_length+1) // 2

        return num_high*num_low - triangle_size


    @classmethod
    def from_archive(cls, archive, **kwargs):
        """Create an InstanceSpec from a CandidateArchive

        This assumes the archive was created based on a MultiFidelityFunction
        with fidelities 'high' and 'low', and that all candidates in the archive
        are available for use.
        All other parameters are simply passed through.
        """

        max_high = archive.count('high')
        max_low = archive.count('low')

        return cls(max_high=max_high, max_low=max_low, **kwargs)
