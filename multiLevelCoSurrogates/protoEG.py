from collections import defaultdict, namedtuple
from typing import Tuple

import numpy as np
from numpy.random import default_rng, Generator
import pandas as pd
import xarray as xr

import multiLevelCoSurrogates as mlcs
from multiLevelCoSurrogates import CandidateArchive, InstanceSpec, MultiFidelityModel
from multiLevelCoSurrogates.Utils import BiFidelityDoE


class ProtoEG:

    def __init__(self, archive: CandidateArchive, rng: Generator, num_reps: int=50):
        """Container for everything needed to create (advanced) Error Grids"""

        self.archive = archive
        self.rng = rng if rng else default_rng()
        self.num_reps = num_reps

        self.models = defaultdict(list)  # models[(n_high, n_low)] = [model_1, ..., model_nreps]
        self.error_grid = None  # xr.DataArray


    def subsample_errorgrid(self):
        """Create an error grid by subsampling from the known archive"""
        instance_spec = InstanceSpec.from_archive(self.archive, num_reps=self.num_reps)

        doe = self.archive.as_doe()

        error_records = []
        for h, l, rep in instance_spec.instances:

            set_seed_by_instance(h, l, r)
            train, test = split_bi_fidelity_doe(doe, h, l)

            train_archive = CandidateArchive(self.archive.ndim, self.archive.fidelities)
            train_archive.addcandidates(..., ..., 'high')
            train_archive.addcandidates(..., ..., 'low')

            model = MultiFidelityModel(fidelities=['high', 'low'], archive=train_archive)
            self.models[(h,l)].append(model)

            # store test

            errors = ...
            error_records.append(errors)

        tmp_df = pd.DataFrame.from_records(error_records)
        self.error_grid = xr.Dataset.from_dataframe(tmp_df)


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
                mlcs.set_seed_by_instance(h, l, idx)
                if fidelity == 'high':
                    selected, left_out = split_bi_fidelity_doe(BiFidelityDoE(high_X, low_X), h-1, l)
                    selected = BiFidelityDoE(np.concatenate([selected.high, X]), selected.low)
                elif fidelity == 'low':
                    selected, left_out = split_bi_fidelity_doe(BiFidelityDoE(high_X, low_X), h, l-1)
                    selected = BiFidelityDoE(selected.high, np.concatenate([selected.low, X]))

                # Create an archive from the MF-function and MF-DoE data
                archive = CandidateArchive(ndim=self.archive.ndim, fidelities=self.archive.fidelities)
                archive.addcandidates(selected.low, low_y, fidelity='low')
                archive.addcandidates(selected.high, high_y, fidelity='high')

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






















