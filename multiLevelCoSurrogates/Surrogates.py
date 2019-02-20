#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Surrogates.py: A generic wrapper for various surrogate models such as Kriging and RBF
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from scipy.interpolate import Rbf

from .Utils import ValueRange, determinerange, rescale


class Surrogate:
    """A generic interface to allow interchangeable use of various models such as RBF, SVM and Kriging"""
    provides_std = False

    def __init__(self, candidate_archive, *,
                 num_points=None, fidelity=None, normalized=True, normalize_target=ValueRange(0, 1)):

        self.archive = candidate_archive
        self.num_points = num_points
        self.fidelity = fidelity
        self.normalized = normalized
        self.normalize_target = normalize_target

        self._surr = None
        self.is_trained = False

        self.X = None
        self.y = None
        self.Xrange = None
        self.yrange = None

        if self.archive is not None and len(self.archive) > 0:
            self._updatevalues()


    def _updatevalues(self):

        X, y = self.archive.getcandidates(num_recent_candidates=self.num_points, fidelity=self.fidelity)

        if self.normalized:
            self.Xrange = determinerange(X)
            self.yrange = determinerange(y)
            X = rescale(X, range_in=self.Xrange, range_out=self.normalize_target)
            y = rescale(y, range_in=self.yrange, range_out=self.normalize_target)

        self.X = X
        self.y = y


    def predict(self, X, *, mode='value', return_std=None):
        """Public prediction function.
        Available modes: 'value', 'std' and 'both'
        """
        if not self.is_trained:
            raise Exception("Cannot predict: surrogate is not trained yet.")

        if return_std is True:
            mode = 'both'
        elif return_std is False:
            mode = 'value'

        if mode == 'value':
            predictor = self.do_predict
        elif mode == 'std':
            predictor = self.do_predict_std
        elif mode == 'both':
            predictor = self.do_predict_both
        else:
            raise ValueError(f"Invalid prediction mode '{mode}'. Supported are: 'value', 'std', 'both'")

        if self.normalized:
            X = rescale(X, range_in=self.Xrange, range_out=self.normalize_target)

        prediction = predictor(X)

        if self.normalized:

            if mode == 'value':
                prediction = rescale(prediction, range_in=self.normalize_target, range_out=self.yrange)
            elif mode == 'std':
                prediction = rescale(prediction, range_in=self.normalize_target, range_out=self.yrange, scale_only=True)
            elif mode == 'both':
                value, std = prediction
                value = rescale(value, range_in=self.normalize_target, range_out=self.yrange)
                std = rescale(std, range_in=self.normalize_target, range_out=self.yrange, scale_only=True)
                prediction = value, std

        return prediction


    def do_predict(self, X):
        raise NotImplementedError

    def do_predict_std(self, X):
        raise NotImplementedError

    def do_predict_both(self, X):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def retrain(self):
        return self.train()


    @classmethod
    def fromname(cls, name, candidate_archive, n=None, fidelity=None, normalized=True, **kwargs):
        if name == 'RBF':
            return RBF(candidate_archive, n, fidelity, normalized=normalized)
        elif name == 'Kriging':
            return Kriging(candidate_archive, n, fidelity, normalized=normalized, **kwargs)
        elif name == 'RandomForest':
            return RandomForest(candidate_archive, n, fidelity, normalized=normalized)
        elif name == 'SVM':
            return SVM(candidate_archive, n, fidelity, normalized=normalized)
        else:
            raise ValueError(f"Unknown surrogate name '{name}'.")


class CoSurrogate:
    """A generic interface for co-surrogates"""

    def __init__(self, surrogate_name, candidate_archive, fidelities, n, fit_scaling_param=True):

        X, y = candidate_archive.getcandidates(num_recent_candidates=n, fidelity=fidelities)
        y_high, y_low = y[:, 0], y[:, 1]

        self.X = X
        self.y_low = np.array(y_low)
        self.y_high = np.array(y_high)

        self.rho = self.determineScaleParameter() if fit_scaling_param else 1
        self.y = y_high - self.rho*y_low

        for idx, x in enumerate(X):
            candidate_archive.addcandidate(x, self.y[idx], fidelity='high-low')

        self.surrogate = Surrogate.fromname(surrogate_name, candidate_archive, n, fidelity='high-low')


    def determineScaleParameter(self):
        """ Determine the scaling parameter 'rho' between y_low and y_high using simple linear regression """
        return 1 / np.mean(self.y_high / self.y_low)


    @property
    def is_trained(self):
        return self.surrogate.is_trained

    @property
    def provides_std(self):
        return self.surrogate.provides_std

    def predict(self, X, *, mode='value'):
        return self.surrogate.predict(X, mode=mode)

    def train(self):
        return self.surrogate.train()


class HierarchicalSurrogate:
    """A generic interface for hierarchical surrogates"""

    def __init__(self, surrogate_name, lower_fidelity_model, candidate_archive, fidelities, *,
                 num_points=None, fit_scaling_param=True, normalized=True, **kwargs):

        self.diff_type = surrogate_name
        self.low_model = lower_fidelity_model
        self.archive = candidate_archive
        self.fidelities = fidelities
        self.n = num_points
        self.fit_scaling_param = fit_scaling_param
        self.normalized = normalized

        self.diff_fidelity = f'{fidelities[0]}-{fidelities[1]}'
        if self.archive is not None and len(self.archive) > 0:
            self.X, self.y_low, self.y_high = self.update_training_values()
            self.rho = self.determineScaleParameter() if fit_scaling_param else 1
            self.y_diff = self.y_high - self.rho * self.y_low
            self.archive.addcandidates(self.X, self.y_diff, fidelity=self.diff_fidelity)
        else:
            self.X = self.y_low = self.y_high = None
            self.rho = None
            self.y_diff = None

        self.diff_model = Surrogate.fromname(surrogate_name, candidate_archive, num_points, fidelity=self.diff_fidelity, normalized=normalized, **kwargs)


    def predict(self, X, *, mode='value', return_std=None):

        if return_std is True:
            mode = 'both'
        elif return_std is False:
            mode = 'value'

        diff_prediction = self.diff_model.predict(X, mode=mode)
        low_prediction = self.low_model.predict(X, mode=mode)

        if mode == 'value':
            diff_value = diff_prediction.reshape((-1,))
            low_value = low_prediction.reshape((-1,))
            diff_std = low_std = 0
        elif mode == 'std':
            diff_std = diff_prediction.reshape((-1,))
            low_std = low_prediction.reshape((-1,))
            diff_value = low_value = 0
        elif mode == 'both':
            diff_value, diff_std = diff_prediction
            diff_value, diff_std = diff_value.reshape((-1,)), diff_std.reshape((-1,))
            low_value, low_std = low_prediction
            low_value, low_std = low_value.reshape((-1,)), low_std.reshape((-1,))
        else:
            raise ValueError(f'Invalid mode {mode}')

        prediction_value = diff_value + self.rho*low_value
        prediction_std = np.sqrt(diff_std**2 + low_std**2)

        if mode == 'value':
            return prediction_value
        elif mode == 'std':
            return prediction_std
        elif mode == 'both':
            return prediction_value, prediction_std


    def train(self):
        self.diff_model.train()
        self.low_model.retrain()


    def retrain(self):
        """Automatically retrain all relevant/required models based on the
        associated CandidateArchive.
        """
        self.X, self.y_low, self.y_high = self.update_training_values()
        self.rho = self.determineScaleParameter() if self.fit_scaling_param else 1
        self.y_diff = self.y_high - self.rho * self.y_low

        self.archive.addcandidates(self.X, self.y_diff, fidelity=self.diff_fidelity)
        self.train()


    def determineScaleParameter(self):
        """ Determine the scaling parameter 'rho' between y_low and y_high using simple linear regression """
        return 1 / np.mean(self.y_high / self.y_low)


    def update_training_values(self):
        assert len(self.fidelities) == 2
        X, y = self.archive.getcandidates(num_recent_candidates=self.n, fidelity=self.fidelities)
        y_high, y_low = y[:, 0], y[:, 1]
        return X,  np.array(y_low), np.array(y_high)


    @property
    def is_trained(self):
        return self.diff_model.is_trained and self.low_model.is_trained

    @property
    def provides_std(self):
        return self.diff_model.provides_std and self.low_model.provides_std



# =============================================================================
# ==================== Specific Surrogates Implementations ====================
# =============================================================================


class RBF(Surrogate):
    """Generic RBF surrogate, implemented by scipy.interpolate.

    Assumes input and output are given as column vectors.
    :param X: input coordinates
    :param y: expected output values
    """
    name = 'RBF'

    def __init__(self, candidate_archive, num_points=None, fidelity=None, normalized=True):
        super(self.__class__, self).__init__(candidate_archive, num_points=num_points, fidelity=fidelity, normalized=normalized)
        self.is_trained = False

    def do_predict(self, X):
        return self._surr(*X.T)

    def train(self):
        self._updatevalues()
        rbf_args = np.hstack((self.X, self.y))
        self._surr = Rbf(*rbf_args.T)
        self.is_trained = True


class Kriging(Surrogate):
    """Generic Kriging surrogate, implemented by sklearn.gaussian_process.GaussianProcessRegressor.

    Assumes input and output are given as column vectors.
    :param X: input coordinates
    :param y: expected output values
    """
    provides_std = True
    name = 'Kriging'

    def __init__(self, candidate_archive, num_points=None, fidelity=None, normalized=True, **kwargs):
        super(self.__class__, self).__init__(candidate_archive, num_points=num_points, fidelity=fidelity, normalized=normalized)
        if 'kernel' not in kwargs:
            kwargs['kernel'] = kernels.RBF()
        self._surr = GaussianProcessRegressor(**kwargs)
        self.is_trained = False

    def train(self):
        self._updatevalues()
        self._surr.fit(self.X, self.y)
        self.is_trained = True

    def set_params(self, **kwargs):
        return self._surr.set_params(**kwargs)

    def do_predict(self, X):
        return self._surr.predict(X).reshape((-1, ))

    def do_predict_std(self, X):
        return self._surr.predict(X, return_std=True)[1]

    def do_predict_both(self, X):
        return self._surr.predict(X, return_std=True)


class RandomForest(Surrogate):
    """Generic Random Forest surrogate, implemented by sklearn.ensemble.RandomForestRegressor.

    Assumes input and output are given as column vectors.
    :param X: input coordinates
    :param y: expected output values
    """
    provides_std = True
    name = 'RandomForest'

    def __init__(self, candidate_archive, num_points=None, fidelity=None, normalized=True):
        super(self.__class__, self).__init__(candidate_archive, num_points=num_points, fidelity=fidelity, normalized=normalized)
        self._surr = RandomForestRegressor(n_estimators=100)
        self.is_trained = False

    def train(self):
        self._updatevalues()
        self._surr.fit(self.X, np.ravel(self.y))
        self.is_trained = True

    def do_predict(self, X):
        return self._surr.predict(X).reshape((-1, ))

    def do_predict_std(self, X):
        stds = []
        for x in X:
            predictions = [est.predict(x.reshape(1, -1))[0] for est in self._surr.estimators_]
            stds.append(np.std(predictions))
        return np.array(stds)

    def do_predict_both(self, X):
        return [self.do_predict(X), self.do_predict_std(X)]


class SVM(Surrogate):
    """Generic SVM regressor surrogate, implemented by sklearn.svm.SVR.

    Assumes input and output are given as column vectors.
    :param X: input coordinates
    :param y: expected output values
    """
    name = 'SVM'

    def __init__(self, candidate_archive, num_points=None, fidelity=None, normalized=True):
        super(self.__class__, self).__init__(candidate_archive, num_points=num_points, fidelity=fidelity, normalized=normalized)
        self._surr = SVR()
        self.is_trained = False

    def do_predict(self, X):
        return self._surr.predict(X).reshape((-1, ))

    def train(self):
        self._updatevalues()
        self._surr.fit(self.X, self.y)
        self.is_trained = True
