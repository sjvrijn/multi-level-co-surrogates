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
from sklearn.linear_model import ElasticNet as EN
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

        self._predictors = {'value': self.do_predict,
                            'std': self.do_predict_std,
                            'both': self.do_predict_both}

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
        """Prediction function.
        Available modes: 'value', 'std' and 'both'
        """
        if not self.is_trained:
            raise Exception("Cannot predict: surrogate is not trained yet.")

        if return_std is True:
            mode = 'both'
        elif return_std is False:
            mode = 'value'

        predictor = self._predictors[mode]

        if self.normalized:
            X = rescale(X, range_in=self.Xrange, range_out=self.normalize_target)

        prediction = predictor(X)

        if self.normalized:

            if mode == 'value':
                prediction = rescale(prediction, range_in=self.normalize_target, range_out=self.yrange).reshape((-1,1))
            elif mode == 'std':
                prediction = rescale(prediction, range_in=self.normalize_target, range_out=self.yrange, scale_only=True).reshape((-1,1))
            elif mode == 'both':
                value, std = prediction
                value = rescale(value, range_in=self.normalize_target, range_out=self.yrange)
                std = rescale(std, range_in=self.normalize_target, range_out=self.yrange, scale_only=True)
                prediction = value.reshape((-1,1)), std.reshape((-1,1))

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
    def fromname(cls, name, candidate_archive, num_points=None, fidelity=None, normalized=True, **kwargs):
        if name == 'RBF':
            return RBF(candidate_archive, num_points, fidelity, normalized=normalized)
        elif name == 'Kriging':
            return Kriging(candidate_archive, num_points, fidelity, normalized=normalized, **kwargs)
        elif name == 'RandomForest':
            return RandomForest(candidate_archive, num_points, fidelity, normalized=normalized)
        elif name == 'SVM':
            return SVM(candidate_archive, num_points, fidelity, normalized=normalized)
        elif name == 'ElasticNet':
            return ElasticNet(candidate_archive, num_points, fidelity, normalized=normalized)
        else:
            raise ValueError(f"Unknown surrogate name '{name}'.")


class HierarchicalSurrogate:
    """A generic interface for hierarchical surrogates"""

    def __init__(self, surrogate_name, lower_fidelity_model, candidate_archive, fidelities, *,
                 num_points=None, scaling='on', normalized=True, **kwargs):

        self.diff_type = surrogate_name
        self.low_model = lower_fidelity_model
        self.archive = candidate_archive
        self.fidelities = fidelities
        self.n = num_points
        self.scaling = scaling
        self.normalized = normalized

        self.diff_fidelity = f'{fidelities[0]}-{fidelities[1]}'
        if self.archive is not None and len(self.archive) > 0:
            self.X, self.y_low, self.y_high = self.update_training_values()
            self.rho = self.determineScaleParameter()
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
        self.rho = self.determineScaleParameter() if self.scaling else 1
        self.y_diff = self.y_high - self.rho * self.y_low

        self.archive.addcandidates(self.X, self.y_diff, fidelity=self.diff_fidelity)
        self.train()


    def determineScaleParameter(self):
        """Determine the scaling parameter 'rho' between y_low and y_high"""
        if self.scaling == 'off':
            rho = 1
        elif self.scaling == 'on':
            rho = 1 / np.mean(self.y_high / self.y_low)
        elif self.scaling == 'inverted':
            rho = 1 / np.mean(self.y_low / self.y_high)
            # print(f'rho = {rho} = 1 / mean({self.y_low/self.y_high})')
        else:
            raise ValueError(f"Scaling option '{self.scaling}' not recognised")

        return rho


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
    """Generic RBF surrogate, implemented by scipy.interpolate."""
    name = 'RBF'

    def __init__(self, candidate_archive, num_points=None, fidelity=None, normalized=True):
        super(self.__class__, self).__init__(candidate_archive, num_points=num_points,
                                             fidelity=fidelity, normalized=normalized)
        self.is_trained = False

    def do_predict(self, X):
        return self._surr(*X.T)

    def train(self):
        self._updatevalues()
        rbf_args = np.hstack((self.X, self.y))
        self._surr = Rbf(*rbf_args.T)
        self.is_trained = True


class Kriging(Surrogate):
    """Generic Kriging surrogate, implemented by
    sklearn.gaussian_process.GaussianProcessRegressor."""
    provides_std = True
    name = 'Kriging'

    def __init__(self, candidate_archive, num_points=None, fidelity=None, normalized=True, **kwargs):
        super(self.__class__, self).__init__(candidate_archive, num_points=num_points,
                                             fidelity=fidelity, normalized=normalized)

        kernel = kwargs.get('kernel', 'RBF')

        if kernel == 'DotProduct':
            kernel = kernels.ConstantKernel(constant_value=1.0) \
                * kernels.DotProduct(sigma_0=1.0)**2
        elif kernel == 'ExpSine':
            kernel = kernels.ConstantKernel(constant_value=1.0) \
                * kernels.ExpSineSquared(length_scale=1.0, periodicity=3.0)
        elif kernel == 'Matern':
            kernel = kernels.ConstantKernel(constant_value=1.0) \
                * kernels.Matern(length_scale=1.0, length_scale_bounds=(1e-5, 5.0))
        elif kernel == 'RationalQuadratic':
            kernel = kernels.ConstantKernel(constant_value=1.0) \
                * kernels.RationalQuadratic(alpha=.1, length_scale=1)
        elif kernel == 'RBF':
            kernel = kernels.ConstantKernel(constant_value=1.0) \
                * kernels.RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

        kwargs['kernel'] = kernel

        self._surr = GaussianProcessRegressor(**kwargs)
        self.is_trained = False

    def train(self):
        self._updatevalues()
        self._surr.fit(self.X, self.y)
        self.is_trained = True

    def set_params(self, **kwargs):
        return self._surr.set_params(**kwargs)

    def do_predict(self, X):
        return self._surr.predict(X).reshape((-1, 1))

    def do_predict_std(self, X):
        return self._surr.predict(X, return_std=True)[1].reshape((-1, 1))

    def do_predict_both(self, X):
        prediction, std = self._surr.predict(X, return_std=True)
        return prediction.reshape((-1, 1)), std.reshape((-1, 1))


class RandomForest(Surrogate):
    """Generic Random Forest surrogate, implemented by
    sklearn.ensemble.RandomForestRegressor."""
    provides_std = True
    name = 'RandomForest'

    def __init__(self, candidate_archive, num_points=None, fidelity=None, normalized=True):
        super(self.__class__, self).__init__(candidate_archive, num_points=num_points,
                                             fidelity=fidelity, normalized=normalized)
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
    """Generic SVM regressor surrogate, implemented by sklearn.svm.SVR."""
    name = 'SVM'

    def __init__(self, candidate_archive, num_points=None, fidelity=None, normalized=True):
        super(self.__class__, self).__init__(candidate_archive, num_points=num_points,
                                             fidelity=fidelity, normalized=normalized)
        self._surr = SVR(gamma='auto')
        self.is_trained = False

    def do_predict(self, X):
        return self._surr.predict(X).reshape((-1, ))

    def train(self):
        self._updatevalues()
        self._surr.fit(self.X, self.y.reshape((-1,)))
        self.is_trained = True


class ElasticNet(Surrogate):
    """Generic Elastic Net regressor surrogate, implemented by
    sklearn.linear_models.ElasticNet."""
    name = 'ElasticNet'

    def __init__(self, candidate_archive, num_points=None, fidelity=None, normalized=True):
        super(self.__class__, self).__init__(candidate_archive, num_points=num_points,
                                             fidelity=fidelity, normalized=normalized)
        self._surr = EN()
        self.is_trained = False

    def do_predict(self, X):
        return self._surr.predict(X).reshape((-1, ))

    def train(self):
        self._updatevalues()
        self._surr.fit(self.X, self.y)
        self.is_trained = True
