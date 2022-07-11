from scipy.stats import norm
from warnings import catch_warnings, simplefilter


class UtilityFunction:
    """
    Code adapted from:
      https://github.com/fmfn/BayesianOptimization/blob/380b0d52ae0e3650b023c4ef6db43f7343c75dea/bayes_opt/util.py
    Under MIT License

    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa=2.576, xi=1, kappa_decay=1, kappa_decay_delay=0):

        self.kappa = kappa
        self.xi = xi

        self._kappa_decay = kappa_decay
        self._kappa_decay_delay = kappa_decay_delay
        self._iters_counter = 0

        if kind not in ['ucb', 'ei', 'ei_orig', 'poi']:
            err = f"The utility function {kind} has not been implemented, " \
                  f"please choose one of ucb, ei, or poi."
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def update_params(self):
        self._iters_counter += 1

        if self._kappa_decay < 1 and self._iters_counter > self._kappa_decay_delay:
            self.kappa *= self._kappa_decay

    def utility(self, x, gp, y_best, goal='maximize'):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa, goal)
        if self.kind == 'ei':
            return self._ei(x, gp, y_best, self.xi, goal)
        if self.kind == 'ei_orig':
            return self._ei_orig(x, gp, y_best, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_best, self.xi, goal)

    @staticmethod
    def _ucb(x, gp, kappa, goal):
        with catch_warnings():
            simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
        if goal == 'maximize':
            return mean + kappa * std
        elif goal == 'minimize':
            return mean - kappa * std

    @staticmethod
    def _ei(x, gp, y_best, xi, goal):
        with catch_warnings():
            simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        mean, std = mean.reshape(-1, 1), std.reshape(-1, 1)

        if goal == 'maximize':
            a = (mean - y_best - xi)
        elif goal == 'minimize':
            a = (y_best - mean + xi)

        z = a / std

        if goal == 'maximize':
            return a * norm.cdf(z) + std * norm.pdf(z)
        elif goal == 'minimize':
            return a * norm.cdf(z) - std * norm.pdf(z)

    @staticmethod
    def _ei_orig(x, gp, y_max, xi):
        with catch_warnings():
            simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)

        a = (mean - y_max - xi)
        z = a / std
        return a * norm.cdf(z) + std * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_best, xi, goal):
        with catch_warnings():
            simplefilter("ignore")
            mean, std = gp.predict(x, return_std=True)
        if goal == 'maximize':
            z = (mean - y_best - xi) / std
        elif goal == 'minimize':
            z = (y_best - mean - xi) / std
        return norm.cdf(z)
