# coding: utf-8
from collections import namedtuple
from itertools import product

import numpy as np
from pathlib import Path
import sys


module_path = Path().joinpath('../..')
if module_path not in sys.path:
    sys.path.append(module_path)

import multiLevelCoSurrogates as mlcs
import multifidelityfunctions as mff

from function_defs import low_lhs_sample, low_random_sample

np.random.seed(20160501)  # Setting seed for reproducibility

np.set_printoptions(linewidth=200)
plot_dir = Path('../../plots/')
file_dir = Path('../../files/')
mlcs.guaranteeFolderExists(plot_dir)
mlcs.guaranteeFolderExists(file_dir)

min_high = 2
min_low = 3
max_high = 50
max_low = 125
num_reps = 50
step = 1


def create_mse_tracking(func, ndim, mfbo_options,
                        max_high=40, max_low=100, num_reps=30,
                        min_high=2, min_low=3, step=1):

    n_test_samples = 500*ndim
    mse_tracking = np.empty((max_high+1, max_low+1, num_reps, 3))
    mse_tracking[:] = np.nan
    r2_tracking = np.empty((max_high+1, max_low+1, num_reps, 3))
    r2_tracking[:] = np.nan
    value_tracking = np.empty((max_high+1, max_low+1, num_reps, 3, n_test_samples))

    test_sample = low_lhs_sample(ndim, n_test_samples)  #TODO: consider rescaling test_sample here instead of in MultiFidBO
    np.save(file_dir.joinpath(f'{ndim}d_test_sample.npy'), test_sample)

    print('starting loops')

    instances = product(range(min_high, max_high+1, step),
                        range(min_low, max_low+1, step),
                        range(num_reps))
    instances = [(h, l, r) for h, l, r in instances if h < l]

    for i, (num_high, num_low, rep) in enumerate(instances):

        if i % 100 == 0:
            print(f'{i}/{len(instances)}')

        np.random.seed(int(f'{num_high:03}{num_low:03}{rep:03}'))

        high_x, low_x = multi_fidelity_doe(ndim, num_high, num_low)

        archive = mlcs.CandidateArchive(ndim=ndim,
                                        fidelities=['high', 'low', 'high-low'])
        archive.addcandidates(low_x, func.low(low_x), fidelity='low')
        archive.addcandidates(high_x, func.high(high_x), fidelity='high')

        mfbo = mlcs.MultiFidelityBO(func, archive, test_sample=test_sample,
                                    **mfbo_options)

        mse_tracking[num_high, num_low, rep] = mfbo.getMSE()
        r2_tracking[num_high, num_low, rep] = mfbo.getR2()

        for i, model in enumerate([mfbo.models['high'],
                                   mfbo.models['low'],
                                   mfbo.direct_models['high']]):

            value_tracking[num_high, num_low, rep, i] = model.predict(mfbo.test_sample).flatten()

    print(f'{len(instances)}/{len(instances)}')
    return mse_tracking, r2_tracking, value_tracking


def multi_fidelity_doe(ndim, num_high, num_low):
    """Create a Design of Experiments (DoE) for two fidelities in `ndim`
    dimensions. The high-fidelity samples are guaranteed to be a subset
    of the low-fidelity samples.

    :returns high-fidelity samples, low-fidelity samples
    """
    low_x = low_lhs_sample(ndim, num_low)
    high_x = low_x[np.random.choice(num_low, num_high, replace=False)]
    return high_x, low_x


if __name__ == '__main__':

    Case = namedtuple('Case', 'ndim func func_name')

    cases = [
        # Case(1, mff.forrester, 'forrester'),

        # Case(2, mff.forrester, 'forrester'),
        Case(2, mff.bohachevsky, 'bohachevsky'),
        Case(2, mff.booth, 'booth'),
        Case(2, mff.branin, 'branin'),
        # Case(2, mff.currin, 'currin'),
        Case(2, mff.himmelblau, 'himmelblau'),
        Case(2, mff.sixHumpCamelBack, 'sixHumpCamelBack'),

        # Case(4, mff.forrester, 'forrester'),
        # Case(4, mff.park91a, 'park91a'),
        Case(4, mff.park91b, 'park91b'),

        Case(6, mff.forrester, 'forrester'),
        Case(6, mff.hartmann6, 'hartmann6'),

        Case(8, mff.forrester, 'forrester'),
        Case(8, mff.borehole, 'borehole'),
    ]

    kernels = ['Matern_']
    scaling_options = ['off']  # , 'on', 'inverted']  # , 'regularized']

    for case, k, scale in product(cases, kernels, scaling_options):

        np.random.seed(20160501)  # Setting seed for reproducibility

        options = {'kernel': k, 'scaling': scale}

        mses, r_squares, values = create_mse_tracking(
            case.func, ndim=case.ndim, mfbo_options=options,
            max_high=max_high, max_low=max_low, num_reps=num_reps,
            min_high=min_high, min_low=min_low, step=step
        )

        base_file_name = f'{k}{case.ndim}d_{case.func_name}'

        np.save(file_dir.joinpath(f'{base_file_name}_lin_mse_tracking.npy'), mses)
        np.save(file_dir.joinpath(f'{base_file_name}_lin_r2_tracking.npy'), r_squares)
        np.save(file_dir.joinpath(f'{base_file_name}_lin_value_tracking.npy'), values)
