from collections import defaultdict

from numpy.random import default_rng, Generator

from multiLevelCoSurrogates import CandidateArchive, Instance, InstanceSpec


class ProtoEG:

    def __init__(self, archive: CandidateArchive, rng: Generator, num_reps: int=50):
        """Container for everything needed to create (advanced) Error Grids"""

        self.archive = archive
        self.rng = rng if rng else default_rng()
        self.num_reps = num_reps

        self.models = defaultdict(list)  # models[(n_high, n_low)] = [model_1, ..., model_nreps]
        self.error_grid = None  # xr.DataArray


    def update_errorgrid_with_sample(self, X, y: float, fidelity: str):
        """Add a new sample of given fidelity and update Error Grid accordingly"""

        instance_spec = InstanceSpec.from_archive(self.archive, num_reps=self.num_reps)
        if fidelity == 'high':
            instance_spec.max_high += 1
        elif fidelity == 'low':
            instance_spec.max_low += 1
        else:
            raise ValueError(f'invalid argument fidelity=`{fidelity}`')

        high_X, high_y = self.archive.getcandidates(fidelity='high')
        low_X, low_y = self.archive.getcandidates(fidelity='low')

        for h, l in instance_spec.pixels:
            fraction = 1 - self.calculate_reuse_fraction(h, l, fidelity)
            num_models_to_resample = fraction * instance_spec.num_reps
            indices_to_resample = self.rng.choice(self.num_reps, size=num_models_to_resample, replace=False)

            for idx in indices_to_resample:
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

        self.archive.addcandidate(X, y, fidelity)

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




























