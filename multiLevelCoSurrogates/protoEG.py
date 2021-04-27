from collections import defaultdict, namedtuple
from typing import Tuple

import numpy as np
from numpy.random import default_rng, Generator
import pandas as pd
from sklearn.metrics import mean_squared_error
import xarray as xr

import multiLevelCoSurrogates as mlcs


class ProtoEG:

    def __init__(self, archive: mlcs.CandidateArchive, rng: Generator=None, num_reps: int=50):
        """Container for everything needed to create (advanced) Error Grids"""

        self.archive = archive
        self.rng = rng  # if rng else default_rng()
        self.num_reps = num_reps

        self.models = defaultdict(list)  # models[(n_high, n_low)] = [model_1, ..., model_nreps]
        self.test_sets = defaultdict(list)  # models[(n_high, n_low)] = [test_1, ..., test_nreps]
        self.error_grid = None  # xr.Dataset


    def subsample_errorgrid(self):
        """Create an error grid by subsampling from the known archive"""
        instance_spec = mlcs.InstanceSpec.from_archive(self.archive, num_reps=self.num_reps)
        instance_spec.max_high -= 1
        doe = self.archive.as_doe()

        error_records = []
        for h, l, rep in instance_spec.instances:

            self.rng = mlcs.set_seed_by_instance(h, l, rep, return_rng=self.rng)
            train, test = split_bi_fidelity_doe(doe, h, l, rng=self.rng)

            test_x = test.high
            self.test_sets[(h, l)].append(test_x)

            train_archive = mlcs.CandidateArchive(self.archive.ndim, self.archive.fidelities)
            train_archive.addcandidates(train.high, self.archive.getfitnesses(train.high, fidelity='high'), 'high')
            train_archive.addcandidates(train.low, self.archive.getfitnesses(train.low, fidelity='low'), 'low')

            model = mlcs.MultiFidelityModel(fidelities=['high', 'low'], archive=train_archive, kernel='Matern', scaling='off')
            self.models[(h,l)].append(model)

            test_y = self.archive.getfitnesses(test_x, fidelity='high')
            mse = mean_squared_error(test_y, model.top_level_model.predict(test_x))
            error_records.append([h, l, rep, 'high_hier', mse])

        columns = ['n_high', 'n_low', 'rep', 'model', 'mses']

        tmp_df = pd.DataFrame.from_records(error_records, columns=columns, index=columns[:4])
        self.error_grid = xr.Dataset.from_dataframe(tmp_df)


    def update_errorgrid_with_sample(self, X, y: float, fidelity: str):
        """Add a new sample of given fidelity and update Error Grid accordingly"""

        instance_spec = mlcs.InstanceSpec.from_archive(self.archive, num_reps=self.num_reps)
        if fidelity == 'high':
            instance_spec.max_high += 1
        elif fidelity == 'low':
            instance_spec.max_low += 1
        else:
            raise ValueError(f'invalid argument fidelity=`{fidelity}`')

        full_DoE = self.archive.as_doe()

        for h, l in instance_spec.pixels:
            fraction = 1 - self.calculate_reuse_fraction(h, l, fidelity)
            num_models_to_resample = int(fraction * instance_spec.num_reps)
            #indices_to_resample = self.rng.choice(self.num_reps, size=num_models_to_resample, replace=False)
            indices_to_resample = np.random.choice(self.num_reps, size=num_models_to_resample, replace=False)

            for idx in indices_to_resample:
                self.rng = mlcs.set_seed_by_instance(h, l, idx)#, return_rng=True)
                if fidelity == 'high':
                    train, test = split_bi_fidelity_doe(full_DoE, h-1, l, rng=self.rng)
                    train = mlcs.BiFidelityDoE(np.concatenate([train.high, X]), train.low)
                    test_high = test.high
                else:  # elif fidelity == 'low':
                    train, test = split_bi_fidelity_doe(full_DoE, h, l-1, rng=self.rng)
                    train = mlcs.BiFidelityDoE(train.high, np.concatenate([train.low, X]))
                    test_high = np.concatenate([test.high, X])

                self.test_sets[(h,l)][idx] = test_high

                # Create an archive from the MF-function and MF-DoE data
                train_archive = mlcs.CandidateArchive(ndim=self.archive.ndim, fidelities=self.archive.fidelities)
                train_archive.addcandidates(train.low, self.archive.getfitnesses(train.low, fidelity='low'), fidelity='low')
                train_archive.addcandidates(train.high, self.archive.getfitnesses(train.high, fidelity='high'), fidelity='high')

                # create and store model
                model = mlcs.MultiFidelityModel(fidelities=['high', 'low'], archive=train_archive,
                                                kernel='Matern', scaling='off')
                self.models[(h, l)][idx] = model

                # calculate and store error of model at this `idx`
                test_y = self.archive.getfitnesses(test_high, fidelity='high')
                mse = mean_squared_error(test_y, model.top_level_model.predict(test_high))
                self.error_grid['mses'].loc[h, l, idx, 'high_hier'] = mse

            if fidelity == 'high':
                indices_to_update_errors = set(range(self.num_reps)) - indices_to_resample
                for idx in indices_to_update_errors:
                    #     add (X, y) to test-set for that model
                    test_high = self.test_sets[(h, l)][idx]
                    test_high = np.concatenate([test_high, X])
                    self.test_sets[(h, l)][idx] = test_high

                    #     recalculate error with new test-set
                    test_y = self.archive.getfitnesses(test_high, fidelity='high')
                    model = self.models[(h,l)][idx]
                    mse = mean_squared_error(test_y, model.top_level_model.predict(test_high))
                    self.error_grid['mses'].loc[h, l, idx, 'high_hier'] = mse

            # elif fidelity == 'low':
                # Error values of remaining models remains unchanged
        #return updated errorgrid (?)

        self.archive.addcandidate(X, y, fidelity)


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



def split_bi_fidelity_doe(DoE: mlcs.BiFidelityDoE, num_high: int, num_low: int,
                          must_include=None, fidelity: str='high',
                          rng: Generator=None) -> Tuple[mlcs.BiFidelityDoE, mlcs.BiFidelityDoE]:
    """Given an existing bi-fidelity Design of Experiments (DoE) `high, low`,
    creates a subselection of given size `num_high, num_low` based on uniform
    selection. The subselection maintains the property that all high-fidelity
    samples are a subset of the low-fidelity samples.

    Raises a `ValueError` if invalid `num_high` or `num_low` are given.
    :param DoE:          Original bi-fidelity DoE to split
    :param num_high:     Number of candidates to select for high-fidelity
    :param num_low:      Number of candidates to select for low-fidelity
    :param must_include: Candidate(s) to explicitly include in 'selected'.
                         Must be an array of shape (num_candidates, ndim).
    :param fidelity:     Which fidelity the 'must_include' candidates should be added as
                         Ignored if nothing given for 'must_include'
    """
    high, low = DoE
    if not 1 < num_high < len(high):
        raise ValueError(f"'num_high' must be in the range [2, len(DoE.high) (={len(DoE.high)})], but is {num_high}")
    elif num_low > len(low):
        raise ValueError(f"'num_low' cannot be greater than len(DoE.low) (={len(DoE.low)}), but is {num_low}")
    elif num_low <= num_high:
        raise ValueError(f"'num_low' must be greater than 'num_high', but {num_low} <= {num_high}")

    must_include_high = must_include and fidelity == 'high'
    must_include_low = must_include and fidelity == 'low'

    rng = rng if rng else np.random  # TODO: replace np.random with np.random.default_rng() (eventually)

    num_high_to_select = num_high - len(must_include) if must_include_high else num_high
    indices = rng.permutation(len(high))
    sub_high, leave_out_high = high[indices[:num_high_to_select]], high[indices[num_high_to_select:]]
    if must_include_high:
        sub_high = np.concatenate([sub_high, must_include])

    num_low_to_select = num_low - len(must_include) if must_include_low else num_low

    if num_low_to_select == len(low):
        sub_low = low
        leave_out_low = []
    else:
        # remove all sub_high from low
        filtered_low = np.array([x for x in low if x not in sub_high])
        # randomly select (num_low - num_high) remaining
        indices = rng.permutation(len(filtered_low))
        num_low_left = num_low_to_select - num_high
        extra_low, leave_out_low = filtered_low[indices[:num_low_left]], \
                                   filtered_low[indices[num_low_left:]]
        # concatenate sub_high with selected sub_low
        sub_low = np.concatenate([sub_high, extra_low], axis=0)
        if must_include_low:
            sub_low = np.concatenate([sub_low, must_include])

    selected = mlcs.BiFidelityDoE(sub_high, sub_low)
    left_out = mlcs.BiFidelityDoE(leave_out_high, leave_out_low)
    return selected, left_out






















