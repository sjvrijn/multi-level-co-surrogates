from collections import defaultdict, namedtuple
from typing import Tuple

import numpy as np
from numpy.random import default_rng, Generator
import pandas as pd
from scipy.special import binom
from sklearn.metrics import mean_squared_error
import xarray as xr

import multiLevelCoSurrogates as mlcs


class ProtoEG:

    def __init__(self, archive: mlcs.CandidateArchive, num_reps: int=50):
        """Container for everything needed to create (advanced) Error Grids"""

        self.archive = archive
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

            mlcs.set_seed_by_instance(h, l, rep)
            train, test = mlcs.split_bi_fidelity_doe(doe, h, l)

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

        # 'full_DoE' should *not* include new sample yet, ...
        full_DoE = self.archive.as_doe()
        # ... but it should be available in self.archive for fitness retrieval
        self.archive.addcandidate(X.flatten(), y, fidelity)
        X = X.reshape(1, -1)

        for h, l in instance_spec.pixels:
            fraction = 1 - self.calculate_reuse_fraction(h, l, fidelity)
            num_models_to_resample = int(fraction * instance_spec.num_reps)
            indices_to_resample = np.random.choice(self.num_reps, size=num_models_to_resample, replace=False)

            for idx in indices_to_resample:
                mlcs.set_seed_by_instance(h, l, idx)
                train, test = mlcs.split_with_include(full_DoE, h, l, must_include=X, fidelity=fidelity)
                test_high = test.high

                self.test_sets[(h,l)][idx] = test_high

                # Create an archive from the MF-function and MF-DoE data
                train_low_y = self.archive.getfitnesses(train.low, fidelity='low')
                train_high_y = self.archive.getfitnesses(train.high, fidelity='high')

                train_archive = mlcs.CandidateArchive(ndim=self.archive.ndim, fidelities=self.archive.fidelities)
                train_archive.addcandidates(train.low, train_low_y, fidelity='low')
                train_archive.addcandidates(train.high, train_high_y, fidelity='high')

                # create and store model
                model = mlcs.MultiFidelityModel(fidelities=['high', 'low'], archive=train_archive,
                                                kernel='Matern', scaling='off')
                self.models[(h,l)][idx] = model

                # calculate and store error of model at this `idx`
                test_y = self.archive.getfitnesses(test_high, fidelity='high')
                mse = mean_squared_error(test_y, model.top_level_model.predict(test_high))
                self.error_grid['mses'].loc[h, l, idx, 'high_hier'] = mse

            if fidelity == 'high':
                indices_to_update_errors = set(range(self.num_reps)) - indices_to_resample
                for idx in indices_to_update_errors:
                    self.update_errors_of_existing_model(X, h, l, idx)

    def update_errors_of_existing_model(self, X, h, l, idx):
        """Add X to test set for models[(h,l)][idx] and recalculate MSE"""
        # add (X, y) to test-set for that model
        test_high = self.test_sets[(h, l)][idx]
        test_high = np.concatenate([test_high, X])
        self.test_sets[(h, l)][idx] = test_high

        # recalculate error with new test-set
        test_y = self.archive.getfitnesses(test_high, fidelity='high')
        model = self.models[(h, l)][idx]
        mse = mean_squared_error(test_y, model.top_level_model.predict(test_high))
        self.error_grid['mses'].loc[h, l, idx, 'high_hier'] = mse

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

        max_high = self.archive.count('high') if not max_high else max_high
        max_low = self.archive.count('low') if not max_low else max_low

        if fidelity == 'high':
            fraction = (max_high-num_high+1) / (max_high+1)
        elif fidelity == 'low':
            fraction = (max_low-num_low+1) / (max_low-num_high+1)
        else:
            raise ValueError(f'Invalid fidelity `{fidelity}` given, expected `high` or `low`.')

        if not (0 <= fraction <= 1):
            raise ValueError('Invalid fraction calculated, please check inputs')
        return fraction
