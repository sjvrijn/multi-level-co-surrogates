#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Surrogates.py: A generic wrapper for various surrogate models such as Kriging and RBF
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from scipy.interpolate import Rbf


def get_min_and_scale(values):
    """Determine the minimum value and scale (max - min) in each dimension"""
    min_vals = np.min(values, axis=0)
    max_vals = np.max(values, axis=0)

    scale = max_vals - min_vals

    return min_vals, scale


def normalize(values, *, min_vals=None, scale=None, target_range=(0.25, 0.75)):
    """Normalize the given values to target range (default: [0,1])"""

    if (min_vals is not None) is not (scale is not None):  # x-or: if just one of them is None
        raise Exception("If specified, both 'min_vals' and 'scale' must be given.")
    elif min_vals is None and scale is None:
        min_vals, scale = get_min_and_scale(values)

    if any(scale == 0):  # Hardcoded error prevention TODO: find better solution?!?
        scale[scale == 0] = 1
        # raise ValueError(f'Scale cannot be 0: {scale}')

    t_min, t_max = target_range

    normalized_values = (values - min_vals) / scale                    # Normalize to [0,1]
    normalized_values = (normalized_values * (t_max - t_min)) + t_min  # Scale to [t_min, t_max]

    return normalized_values


def denormalize(values, min_vals, scale, *, target_range=(0.25, 0.75)):
    """Denormalize the given normalized values, default assumed normalization target is [0,1]"""

    t_min, t_max = target_range

    denormalized_values = (values - t_min) / (t_max - t_min)         # Reverse [t_min, t_max] to [0,1] scale
    denormalized_values = (denormalized_values * scale) + min_vals   # Denormalize to original range

    return denormalized_values


class Surrogate:
    """A generic interface to allow interchangeable use of various models such as RBF, SVM and Kriging"""
    provides_std = False

    def __init__(self, candidate_archive, n, *, fidelity=None, normalized=True, normalize_target=(0.25, 0.75)):
        X, y = candidate_archive.getcandidates(n=n, fidelity=fidelity)

        self._surr = None

        self.normalized = normalized
        if normalized:
            self.normalize_target = normalize_target
            self.Xmins, self.Xscales = get_min_and_scale(X)
            self.ymin, self.yscale = get_min_and_scale(y)
            X = normalize(X, target_range=normalize_target)
            y = normalize(y, target_range=normalize_target)

        self.X = X
        self.y = y
        self.is_trained = False

    def predict(self, X, *, mode='value'):
        """Public prediction function. Available modes: 'value' and 'std'"""
        if not self.is_trained:
            raise Exception("Cannot predict: surrogate is not trained yet.")

        if mode == 'value':
            predictor = self.do_predict
        elif mode == 'std':
            predictor = self.do_predict_std
        else:
            raise ValueError(f"Invalid prediction mode '{mode}'. Supported are: 'value', 'std'")

        if self.normalized:
            X = normalize(X, min_vals=self.Xmins, scale=self.Xscales, target_range=self.normalize_target)

        prediction = predictor(X)

        if self.normalized and mode == 'value':
            prediction = denormalize(prediction, min_vals=self.ymin, scale=self.yscale,
                                     target_range=self.normalize_target)

        return prediction


    def do_predict(self, X):
        raise NotImplementedError

    def do_predict_std(self, X):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError


    @classmethod
    def fromname(cls, name, candidate_archive, n, fidelity=None):
        if name == 'RBF':
            return RBF(candidate_archive, n, fidelity)
        elif name == 'Kriging':
            return Kriging(candidate_archive, n, fidelity)
        elif name == 'RandomForest':
            return RandomForest(candidate_archive, n, fidelity)
        elif name == 'SVM':
            return SVM(candidate_archive, n, fidelity)
        else:
            raise ValueError(f"Unknown surrogate name '{name}'.")


class CoSurrogate:
    """A generic interface for co-surrogates"""

    def __init__(self, surrogate_name, candidate_archive, fidelities, n, fit_scaling_param=True):

        X, y = candidate_archive.getcandidates(n=n, fidelity=fidelities)
        y_high, y_low = y[:, 0], y[:, 1]

        self.X = X
        self.y_low = np.array(y_low)
        self.y_high = np.array(y_high)

        self.rho = self.determineScaleParameter() if fit_scaling_param else 1
        self.y = y_high - self.rho*y_low

        for idx, x in enumerate(X):
            candidate_archive.updatecandidate(x, self.y[idx], fidelity='high-low')

        self.surrogate = Surrogate.fromname(surrogate_name, candidate_archive, n, fidelity='high-low')


    def determineScaleParameter(self):
        """ Determine the scaling parameter 'rho' between y_low and y_high using simple linear regression """
        regr = LinearRegression()
        regr.fit(self.y_low.reshape(-1, 1), self.y_high.reshape(-1, 1))
        return regr.coef_.flatten()[0]


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


class RBF(Surrogate):
    """Generic RBF surrogate, implemented by scipy.interpolate.

    Assumes input and output are given as column vectors.
    :param X: input coordinates
    :param y: expected output values
    """
    name = 'RBF'

    def __init__(self, candidate_archive, n, fidelity=None):
        super(self.__class__, self).__init__(candidate_archive, n, fidelity=fidelity)
        self.is_trained = False

    def do_predict(self, X):
        return self._surr(*X.T)

    def train(self):
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

    def __init__(self, candidate_archive, n, fidelity=None):
        super(self.__class__, self).__init__(candidate_archive, n, fidelity=fidelity)
        self._surr = GaussianProcessRegressor()
        self.is_trained = False

    def do_predict(self, X):
        return self._surr.predict(X).reshape((-1, ))

    def train(self):
        self._surr.fit(self.X, self.y)
        self.is_trained = True

    def do_predict_std(self, X):
        return self._surr.predict(X, return_std=True)[1]


class RandomForest(Surrogate):
    """Generic Random Forest surrogate, implemented by sklearn.ensemble.RandomForestRegressor.

    Assumes input and output are given as column vectors.
    :param X: input coordinates
    :param y: expected output values
    """
    provides_std = True
    name = 'RandomForest'

    def __init__(self, candidate_archive, n, fidelity=None):
        super(self.__class__, self).__init__(candidate_archive, n, fidelity=fidelity)
        self._surr = RandomForestRegressor()
        self.is_trained = False

    def do_predict(self, X):
        return self._surr.predict(X).reshape((-1, ))

    def train(self):
        self._surr.fit(self.X, np.ravel(self.y))
        self.is_trained = True

    def do_predict_std(self, X):
        stds = []
        for x in X:
            predictions = [est.predict(x.reshape(1, -1))[0] for est in self._surr.estimators_]
            stds.append(np.std(predictions))
        return np.array(stds)


class SVM(Surrogate):
    """Generic SVM regressor surrogate, implemented by sklearn.svm.SVR.

    Assumes input and output are given as column vectors.
    :param X: input coordinates
    :param y: expected output values
    """
    name = 'SVM'

    def __init__(self, candidate_archive, n, fidelity=None):
        super(self.__class__, self).__init__(candidate_archive, n, fidelity=fidelity)
        self._surr = SVR()
        self.is_trained = False

    def do_predict(self, X):
        return self._surr.predict(X).reshape((-1, ))

    def train(self):
        self._surr.fit(self.X, self.y)
        self.is_trained = True