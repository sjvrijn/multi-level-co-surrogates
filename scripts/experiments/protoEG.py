from collections import defaultdict, namedtuple
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

    def __init__(self, archive: CandidateArchive):
        """Container for everything needed to create (advanced) Error Grids"""

        self.archive = archive
        self.models = defaultdict(list)  # models[(n_high, n_low)] = [model_1, ..., model_nreps]
        self.error_grid = None  # xr.DataArray


    def add_new_sample(self, X, y: float, fidelity: str):
        """Add a new sample of given fidelity and update Error Grid accordingly"""

        #Add (X, y) to archive
        #
        #
        #for pixel in error grid:
        #
        #    fraction = 1 - calculate_resample_fraction(h, l, fidelity)
        #
        #    num_models_to_resample = fraction * num_reps
        #    uniform at random select num_models_to_resample from num_reps
        #
        #    for each model_to_resample:
        #        create new subsample consisting of at least latest addition X
        #        create and store model
        #        calculate and store error
        #
        #    if fidelity == 'low':
        #        # Error values of remaining models remains unchanged
        #    elif fidelity == 'high':
        #        for each model_not_resampled:
        #            add (X, y) to test-set for that model
        #            recalculate error with new test-set
        #
        #return updated errorgrid (?)
        raise NotImplementedError()


    def calculate_reuse_fraction(self, num_high: int, num_low: int, fidelity: str,
                                 *, max_high: int=None, max_low: int=None) -> float:
        """Calculate the fraction of models that can be reused

        Given `max_high` H, `max_low` L, `num_high` h and `num_low` l, the number of
        unique possible subsamples is given by binom(H, h) * binom(L-h, l-h), i.e.:

        /H\/L-h\
        \h/\l-h/

        In the iterative case when samples are added to H or L one at a time,
        it can be expected that some of the subsamples would not use the new
        samples, and therefore that a fraction of previous subsamples can be
        reused when calculating the Error Grid for the next iteration.

        Assuming subsampling is done uniformly at random, the fraction of
        subsamples in the 'next' iteration that only use samples from the
        previous iteration is equal to the ratio between the numbers of total
        possible subsamples for those given sizes, i.e.:

        /H\/L-h\
        \h/\l-h/
        ___________

        /H+1\/L-h\
        \ h /\l-h/

        if H := H+1, or


        /H\/L-h\
        \h/\l-h/
        ___________

        /H\/L+1-h\
        \h/\ l-h /

        if L := L+1

        :param num_high: number of high-fidelity samples in the subsample
        :param num_low: number of low-fidelity samples in the subsample
        :param fidelity: fidelity in which the latest sample has been added
        :param max_high: total number of high-fidelity samples, taken from self.archive if not given
        :param max_low: total number of low-fidelity samples, taken from self.archive if not given
        :returns: fraction [0, 1] of samples that can be reused
        """

        max_high = len(self.archive.getcandidates(fidelity='high')) if not max_high else max_high
        max_low = len(self.archive.getcandidates(fidelity='low')) if not max_low else max_low

        if fidelity == 'high':
            return (max_high-num_high+1) / (max_high+1)

        if fidelity == 'low':
            return (max_low-num_low+1) / (max_low-num_high+1)

        raise ValueError(f'Invalid fidelity `{fidelity}` given, expected `high` or `low`.')




























