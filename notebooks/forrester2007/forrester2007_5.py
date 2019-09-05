# coding: utf-8


from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# from numba import jit, prange

module_path = Path().joinpath('../..')
if module_path not in sys.path:
    sys.path.append(module_path)

from itertools import product
import multifidelityfunctions as mff
import multiLevelCoSurrogates as mlcs

from function_defs import low_lhs_sample, low_random_sample, TD_inv
np.random.seed(20160501)  # Setting seed for reproducibility
OD = mff.oneDimensional


np.set_printoptions(linewidth=200)
plot_dir = '../../plots/'
file_dir = '../../files/'


min_high = 2
min_low = 3
max_high = 50
max_low = 125
num_reps = 50
step = 1

# plot_x = np.linspace(0, 1, 101).reshape(-1, 1)


#@jit(parallel=True)
def create_mse_tracking(func, sample_generator, gp_kernel='',
                        max_high=40, max_low=100, num_reps=30,
                        min_high=2, min_low=3, step=1, scaling='on'):
    ndim = func.ndim
    n_test_samples = 1000
    mse_tracking = np.empty((max_high+1, max_low+1, num_reps, 3))
    mse_tracking[:] = np.nan
    r2_tracking = np.empty((max_high + 1, max_low + 1, num_reps, 3))
    r2_tracking[:] = np.nan

    value_tracking = np.empty((max_high+1, max_low+1, num_reps, 3, n_test_samples))

    cases = list(product(range(min_high, max_high+1, step), range(min_low, max_low+1, step), range(num_reps)))


    input_range = mlcs.ValueRange(*np.array([func.l_bound, func.u_bound], dtype=np.float))
    output_range = (-10, 16)
    test_sample = mlcs.sample_by_function(func.high, n_samples=n_test_samples, ndim=func.ndim,
                                          range_in=input_range, range_out=output_range, minimize=False)

    np.save(f'{ndim}d_test_sample.npy', test_sample)

    print('starting loops')

    # bad_MSE = []
    # bad_R2 = []

    #for idx, case in enumerate(cases):
    #    num_high, num_low, rep = case
    for num_high in range(min_high, max_high+1, step):
        for num_low in range(min_low, max_low+1, step):
            for rep in range(num_reps):

                idx = (num_high-min_high)//step * num_reps * (max_low-min_low)//step \
                        + (num_low-min_low)//step * num_reps \
                        + rep

                idx = int(idx)

                if idx % 100 == 0:
                    print(idx, '/', len(cases))
                if num_high >= num_low:
                    continue

                low_x = sample_generator(ndim, num_low)
                high_x = low_x[np.random.choice(num_low, num_high, replace=False)]
        
                archive = mlcs.CandidateArchive(ndim=ndim, fidelities=['high', 'low', 'high-low'])
                archive.addcandidates(low_x, func.low(low_x), fidelity='low')
                archive.addcandidates(high_x, func.high(high_x), fidelity='high')

                # print(idx)
                mfbo = mlcs.MultiFidelityBO(func, archive, output_range=output_range, test_sample=test_sample,
                                            kernel=gp_kernel[:-1], scaling=scaling)

                MSEs = mfbo.getMSE()
                R2s = mfbo.getR2()

                # if MSEs.high_hier > MSEs.high:
                #     bad_MSE.append(idx)
                # if R2s.high_hier < R2s.high:
                #     bad_R2.append(idx)

                mse_tracking[num_high, num_low, rep] = MSEs
                r2_tracking[num_high, num_low, rep] = R2s

                # plt.plot(plot_x, func.high(plot_x), label='true function')
                # plt.plot(plot_x, mfbo.models['high'].predict(plot_x), label='hierarchical model')
                # plt.plot(plot_x, mfbo.direct_models['high'].predict(plot_x), label='direct model')
                # plt.scatter(*archive.getcandidates(fidelity='low'), label='low fidelity samples', s=42, zorder=4, marker='o', facecolors='none', color='red')
                # plt.scatter(*archive.getcandidates(fidelity='high'), label='high fidelity samples', s=42, zorder=5, marker='x', color='black')
                # plt.legend()
                # plt.title(f'{num_high} from {num_low} samples\n'
                #           f'(MSE: {np.round(MSEs.high_hier, 2)} vs {np.round(MSEs.high, 2)})\n'
                #           f'(R^2: {np.round(R2s.high_hier, 2)} vs {np.round(R2s.high, 2)})')
                # plt.tight_layout()
                # plt.savefig(f'{plot_dir}{gp_kernel}case_{idx}_{num_low}choose{num_high}.png')
                # plt.clf()

                for i, (direct, fid) in enumerate([(False, 'high'), (False, 'low'), (True, 'high')]):
                    if direct:
                        models = mfbo.direct_models
                    else:
                        models = mfbo.models

                    value_tracking[num_high, num_low, rep, i] = models[fid].predict(mfbo.test_sample).flatten()

    # with open(f'{plot_dir}{gp_kernel}num_bad.txt', 'w') as f:
    #     f.write(f'#Bad cases by MSE: {len(bad_MSE)}\n'
    #             f'#Bad cases by R^2: {len(bad_R2)}\n'
    #             f'\n'
    #             f'Bad MSE indices: {bad_MSE}\n'
    #             f'Bad R^2 indices: {bad_R2}\n')

    print(len(cases), '/', len(cases))
    return mse_tracking, r2_tracking, value_tracking




# kernels = ['DotProduct_', 'ExpSine_', 'Matern_', 'RationalQuadratic_', 'RBF_']
kernels = ['Matern_', 'RationalQuadratic_', 'RBF_']

# # EGO - 1D function
inv_OD = mff.MultiFidelityFunction(
    u_bound=np.array(OD.u_bound), l_bound=np.array(OD.l_bound),
    functions=[lambda x: -OD.high(x), lambda x: -OD.low(x)],
    fidelity_names=['high', 'low'],
)


# base_plot_dir = plot_dir
scaling_options = ['off']  # , 'on', 'inverted']  # , 'regularized']
# subdirs = ['disabled_scaling/', 'original_scaling/', 'inverted_scaling/']  # , '3_regularized_scaling/']

for k in kernels:
    # for scale, subdir in zip(scaling_options, subdirs):
    #     plot_dir = base_plot_dir + subdir
    for scale in scaling_options:
        np.random.seed(20160501)  # Setting seed for reproducibility
        mse_tracking, r2_tracking, values = create_mse_tracking(inv_OD, low_lhs_sample, gp_kernel=k,
                                                                max_high=max_high, max_low=max_low,
                                                                num_reps=num_reps, min_high=min_high,
                                                                min_low=min_low, step=step,
                                                                scaling=scale)
        np.save(f'{file_dir}{k}1d_lin_mse_tracking.npy', mse_tracking)
        np.save(f'{file_dir}{k}1d_lin_r2_tracking.npy', r2_tracking)
        np.save(f'{file_dir}{k}1d_lin_value_tracking.npy', values)


for k in kernels:
    # for scale, subdir in zip(scaling_options, subdirs):
    for scale in scaling_options:
        np.random.seed(20160501)  # Setting seed for reproducibility
        mse_tracking, r2_tracking, values = create_mse_tracking(TD_inv, low_lhs_sample, gp_kernel=k,
                                                                max_high=max_high, max_low=max_low,
                                                                num_reps=num_reps, min_high=min_high,
                                                                min_low=min_low, step=step,
                                                                scaling=scale)

        np.save(f'{file_dir}{k}2d_lin_mse_tracking.npy', mse_tracking)
        np.save(f'{file_dir}{k}2d_lin_r2_tracking.npy', r2_tracking)
        np.save(f'{file_dir}{k}2d_lin_value_tracking.npy', values)



if __name__ == '__main__':
    pass
