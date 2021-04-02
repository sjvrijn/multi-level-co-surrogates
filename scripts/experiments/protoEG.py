from collections import namedtuple
from dataclasses import dataclass
from itertools import product
from warnings import warn


Instance = namedtuple('Instance', 'n_high n_low rep')


@dataclass
class InstanceSpec:
    min_high:  int = 2
    min_low:   int = 3
    max_high:  int
    max_low:   int
    step:      int = 1
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

        self.instances = [
            Instance(h, l, r)
            for h, l, r in product(range(min_high, max_high + 1, step_high),
                                   range(min_low, max_low + 1, step_low),
                                   range(num_reps))
            if h < l
        ]


class ProtoEG:

    def __init__(self):
        """Container for everything needed to create (advanced) Error Grids

        Specifically contains everything needed to
        """
        pass


