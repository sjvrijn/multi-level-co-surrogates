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
from scipy.interpolate import Rbf


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


class RBF(Surrogate):
    """Generic RBF surrogate, implemented by scipy.interpolate.

    Assumes input and output are given as column vectors.
    :param X: input coordinates
    :param y: expected output values
    """

    def __init__(self, X, y):
        super(self.__class__, self).__init__(X, y)
        self.name = 'RBF'
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

    def __init__(self, X, y):
        super(self.__class__, self).__init__(X, y)
        self._surr = GaussianProcessRegressor()
        self.name = 'Kriging'
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

    def __init__(self, X, y):
        super(self.__class__, self).__init__(X, y)
        self._surr = RandomForestRegressor()
        self.name = 'RandomForest'
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
        return stds
