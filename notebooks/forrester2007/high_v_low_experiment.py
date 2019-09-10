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

from function_defs import low_lhs_sample

np.random.seed(20160501)  # Setting seed for reproducibility

np.set_printoptions(linewidth=200)
plot_dir = Path('../../plots/')
file_dir = Path('../../files/')
mlcs.guaranteeFolderExists(plot_dir)
mlcs.guaranteeFolderExists(file_dir)


Case = namedtuple('Case', 'ndim func func_name')

def create_mse_tracking(func, ndim, mfbo_options, instances):

    n_test_samples = 500*ndim
    mse_tracking = np.empty((len(instances), 3))
    mse_tracking[:] = np.nan
    r2_tracking = np.empty((len(instances), 3))
    r2_tracking[:] = np.nan
    value_tracking = np.empty((len(instances), 3, n_test_samples))

    test_sample = low_lhs_sample(ndim, n_test_samples)  #TODO: consider rescaling test_sample here instead of in MultiFidBO
    np.save(file_dir.joinpath(f'{ndim}d_test_sample.npy'), test_sample)
    mfbo_options['test_sample'] = test_sample

    print('starting loops')

    for i, instance in enumerate(instances):

        if i % 100 == 0:
            print(f'{i}/{len(instances)}')

        mfbo = create_experiment_instance(func, mfbo_options, ndim, instance)

        mse_tracking[i] = mfbo.getMSE()
        r2_tracking[i] = mfbo.getR2()

        for m, model in enumerate([mfbo.models['high'],
                                   mfbo.models['low'],
                                   mfbo.direct_models['high']]):
            value_tracking[i, m] = model.predict(mfbo.test_sample).flatten()

    print(f'{len(instances)}/{len(instances)}')
    return mse_tracking, r2_tracking, value_tracking


def create_experiment_instance(func, mfbo_options, ndim, instance):
    """Create a consistent instantiated MFBO instance with the given parameters.

    :returns MultiFidelityBO instance
    """
    num_high, num_low, rep = instance

    np.random.seed(int(f'{num_high:03}{num_low:03}{rep:03}'))

    high_x, low_x = multi_fidelity_doe(ndim, num_high, num_low)

    archive = mlcs.CandidateArchive(ndim=ndim,
                                    fidelities=['high', 'low', 'high-low'])
    archive.addcandidates(low_x, func.low(low_x), fidelity='low')
    archive.addcandidates(high_x, func.high(high_x), fidelity='high')

    mfbo = mlcs.MultiFidelityBO(func, archive, **mfbo_options)

    return mfbo


def multi_fidelity_doe(ndim, num_high, num_low):
    """Create a Design of Experiments (DoE) for two fidelities in `ndim`
    dimensions. The high-fidelity samples are guaranteed to be a subset
    of the low-fidelity samples.

    :returns high-fidelity samples, low-fidelity samples
    """
    low_x = low_lhs_sample(ndim, num_low)
    high_x = low_x[np.random.choice(num_low, num_high, replace=False)]
    return high_x, low_x


def run(cases, kernels, scaling_options, instances):

    for case, k, scale in product(cases, kernels, scaling_options):

        np.random.seed(20160501)  # Setting seed for reproducibility

        options = {'kernel': k[:-1], 'scaling': scale}

        mses, r_squares, values = \
            create_mse_tracking(func=case.func, mfbo_options=options,
                                ndim=case.ndim, instances=instances)

        base_file_name = f'{k}{case.ndim}d_{case.func_name}'

        # TODO: store as Pandas DataFrame/XArray DataArray/DataSet instead
        np.save(file_dir.joinpath(f'{base_file_name}_instances.npy'), np.array(instances))
        np.save(file_dir.joinpath(f'{base_file_name}_lin_mse_tracking.npy'), mses)
        np.save(file_dir.joinpath(f'{base_file_name}_lin_r2_tracking.npy'), r_squares)
        np.save(file_dir.joinpath(f'{base_file_name}_lin_value_tracking.npy'), values)
