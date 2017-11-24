#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

'''
Surrogates.py: A generic wrapper for various surrogate models such as Kriging and RBF
'''

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy.interpolate import Rbf


def get_min_and_scale(values):
    """Determine the minimum value and scale (max - min) in each dimension"""
    min_vals = np.min(values, axis=0)
    max_vals = np.max(values, axis=0)

    scale = max_vals - min_vals

    return min_vals, scale


def normalize(values, *, min_vals=None, scale=None, target_range=(0,1)):
    """Normalize the given values to target range (default: [0,1])"""

    if (min_vals is not None) is not (scale is not None):  # x-or: if just one of them is None
        raise Exception("If specified, both 'min_vals' and 'scale' must be given.")
    elif min_vals is None and scale is None:
        min_vals, scale = get_min_and_scale(values)

    t_min, t_max = target_range

    normalized_values = (values - min_vals) / scale                    # Normalize to [0,1]
    normalized_values = (normalized_values * (t_max - t_min)) + t_min  # Scale to [t_min, t_max]

    return normalized_values


def denormalize(values, min_vals, scale, *, target_range=(0,1)):
    """Denormalize the given normalized values, default assumed normalization target is [0,1]"""

    t_min, t_max = target_range

    denormalized_values = (values - t_min) / (t_max - t_min)         # Reverse [t_min, t_max] to [0,1] scale
    denormalized_values = (denormalized_values * scale) + min_vals   # Denormalize to original range

    return denormalized_values


class Surrogate:
    """A generic interface to allow interchangeable use of various models such as RBF, SVM and Kriging"""
    provides_std = False

    def __init__(self, X, y):
        self._surr = None
        self.X = X
        self.y = y
        self.is_trained = False

    def predict(self, X):
        if self.is_trained:
            return self.do_predict(X)
        else:
            raise Exception("Cannot predict: surrogate is not trained yet.")

    def do_predict(self, X):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def addPoint(self, point):
        raise NotImplementedError

    def addPoints(self, points):
        for point in points:
            self.addPoint(point)

    @classmethod
    def fromname(cls, name, X, y):
        if name == 'RBF':
            return RBF(X, y)
        elif name == 'Kriging':
            return Kriging(X, y)
        elif name == 'RandomForest':
            return RandomForest(X, y)
        else:
            raise ValueError(f"Unknown surrogate name '{name}'.")


class CoSurrogate:
    """A generic interface for co-surrogates"""

    def __init__(self, surrogate_name, X, y_low, y_high, fit_scaling_param=True):

        self.X = X
        self.y_low = np.array(y_low)
        self.y_high = np.array(y_high)

        self.rho = self.determineScaleParameter() if fit_scaling_param else 1
        self.y = y_high - self.rho*y_low

        self.surrogate = Surrogate.fromname(surrogate_name, X, self.y)


    def determineScaleParameter(self):
        """ Determine the scaling parameter 'rho' between y_low and y_high using simple linear regression """
        regr = LinearRegression()
        regr.fit(self.y_low, self.y_high)
        return regr.coef_.flatten()[0]


    @property
    def is_trained(self):
        return self.surrogate.is_trained

    @property
    def provides_std(self):
        return self.surrogate.provides_std

    def predict(self, X):
        return self.surrogate.predict(X)

    def predict_std(self, X):
        return self.surrogate.predict_std(X)

    def train(self):
        return self.surrogate.train()


class RBF(Surrogate):
    """Generic RBF surrogate, implemented by scipy.interpolate.

    Assumes input and output are given as column vectors.
    :param X: input coordinates
    :param y: expected output values
    """
    name = 'RBF'

    def __init__(self, X, y):
        super(self.__class__, self).__init__(X, y)
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

    def __init__(self, X, y):
        super(self.__class__, self).__init__(X, y)
        self._surr = GaussianProcessRegressor()
        self.is_trained = False

    def do_predict(self, X):
        return self._surr.predict(X).reshape((-1, ))

    def train(self):
        self._surr.fit(self.X, self.y)
        self.is_trained = True

    def predict_std(self, X):
        return self._surr.predict(X, return_std=True)[1]


class RandomForest(Surrogate):
    """Generic Random Forest surrogate, implemented by sklearn.ensemble.RandomForestRegressor.

    Assumes input and output are given as column vectors.
    :param X: input coordinates
    :param y: expected output values
    """
    provides_std = True
    name = 'RandomForest'

    def __init__(self, X, y):
        super(self.__class__, self).__init__(X, y)
        self._surr = RandomForestRegressor()
        self.is_trained = False

    def do_predict(self, X):
        return self._surr.predict(X).reshape((-1, ))

    def train(self):
        self._surr.fit(self.X, np.ravel(self.y))
        self.is_trained = True

    def predict_std(self, X):
        stds = []
        for x in X:
            predictions = [est.predict(x.reshape(1, -1))[0] for est in self._surr.estimators_]
            stds.append(np.std(predictions))
        return np.array(stds)
