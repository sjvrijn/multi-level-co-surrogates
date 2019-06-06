# coding: utf-8
import numpy as np
import os
import sys


module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import multiLevelCoSurrogates as mlcs

from function_defs import low_lhs_sample, low_random_sample, forrester

np.random.seed(20160501)  # Setting seed for reproducibility

np.set_printoptions(linewidth=200)
plot_dir = '../../plots/'
file_dir = '../../files/'
mlcs.guaranteeFolderExists(plot_dir)
mlcs.guaranteeFolderExists(file_dir)

min_high = 2
min_low = 3
max_high = 50
max_low = 125
num_reps = 30
step = 3


# @jit(parallel=True)
def create_mse_tracking(func, sample_generator, ndim, gp_kernel='',
                        max_high=40, max_low=100, num_reps=30,
                        min_high=2, min_low=3, step=1, scaling='on'):

    n_test_samples = 1000
    mse_tracking = np.empty((max_high+1, max_low+1, num_reps, 3))
    mse_tracking[:] = np.nan
    r2_tracking = np.empty((max_high+1, max_low+1, num_reps, 3))
    r2_tracking[:] = np.nan

    value_tracking = np.empty((max_high+1, max_low+1, num_reps, 3, n_test_samples))

    num_cases = (max_high-min_high)//step * (max_low-min_low)//step * num_reps

    input_range = mlcs.ValueRange(*np.array([func.l_bound, func.u_bound], dtype=np.float))
    output_range = (0, 22*ndim)
    test_sample = mlcs.sample_by_function(func.high, n_samples=n_test_samples, ndim=ndim,
                                          range_in=input_range, range_out=output_range, minimize=False)

    np.save(f'{ndim}d_test_sample.npy', test_sample)

    print('starting loops')

    for num_high in range(min_high, max_high+1, step):
        for num_low in range(min_low, max_low+1, step):
            for rep in range(num_reps):

                idx = (num_high - min_high) // step * num_reps * (max_low - min_low) // step \
                      + (num_low - min_low) // step * num_reps \
                      + rep

                idx = int(idx)

                if idx % 100 == 0:
                    print(idx, '/', num_cases)
                if num_high >= num_low:
                    continue

                low_x = sample_generator(ndim, num_low)
                high_x = low_x[np.random.choice(num_low, num_high, replace=False)]

                archive = mlcs.CandidateArchive(ndim=ndim, fidelities=['high', 'low', 'high-low'])
                archive.addcandidates(low_x, func.low(low_x), fidelity='low')
                archive.addcandidates(high_x, func.high(high_x), fidelity='high')

                mfbo = mlcs.MultiFidelityBO(func, archive, output_range=output_range, test_sample=test_sample,
                                            kernel=gp_kernel[:-1], scaling=scaling)

                MSEs = mfbo.getMSE()
                R2s = mfbo.getR2()


                mse_tracking[num_high, num_low, rep] = MSEs
                r2_tracking[num_high, num_low, rep] = R2s

                for i, (direct, fid) in enumerate([(False, 'high'), (False, 'low'), (True, 'high')]):
                    if direct:
                        models = mfbo.direct_models
                    else:
                        models = mfbo.models

                    value_tracking[num_high, num_low, rep, i] = models[fid].predict(mfbo.test_sample).flatten()


    print(num_cases, '/', num_cases)
    return mse_tracking, r2_tracking, value_tracking


if __name__ == '__main__':
    ndim = int(input("Dimensionality: "))

    kernels = ['Matern_']
    scaling_options = ['off']  # , 'on', 'inverted']  # , 'regularized']

    for k in kernels:
        for scale in scaling_options:
            np.random.seed(20160501)  # Setting seed for reproducibility
            mse_tracking, r2_tracking, values = create_mse_tracking(
                forrester, low_lhs_sample, ndim=ndim, gp_kernel=k,
                max_high=max_high, max_low=max_low, num_reps=num_reps,
                min_high=min_high, min_low=min_low, step=step, scaling=scale
            )

            np.save(f'{file_dir}{k}{ndim}d_lin_mse_tracking.npy', mse_tracking)
            np.save(f'{file_dir}{k}{ndim}d_lin_r2_tracking.npy', r2_tracking)
            np.save(f'{file_dir}{k}{ndim}d_lin_value_tracking.npy', values)
