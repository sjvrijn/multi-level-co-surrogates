# coding: utf-8
import numpy as np
import os
import sys


module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import multiLevelCoSurrogates as mlcs
import multifidelityfunctions as mff


import math
import matplotlib.pyplot as plt
import more_itertools
import pyDOE

plot_dir = 'plots/'
mlcs.guaranteeFolderExists(plot_dir)

np.set_printoptions(precision=5, linewidth=180)



np.random.seed(20160501)  # Setting seed for reproducibility

np.set_printoptions(linewidth=200)
plot_dir = '../../plots/'
file_dir = '../../files/'
mlcs.guaranteeFolderExists(plot_dir)
mlcs.guaranteeFolderExists(file_dir)


# Some basic initial setup
func = mff.forrester_high
lboundx, uboundx = 0, 1
lboundy, uboundy = 0, 22

ndim = 1
n_test = 100
n_steps = 25
n_reps = 1
n_init = 5


def iterative_EGO_sampling(ndim):
    init_x = np.random.rand(n_init, ndim).reshape(-1, ndim)
    init_x = mlcs.rescale(init_x, range_in=(0, 1), range_out=(lboundx, uboundx))

    test_x_rand = np.random.rand(n_test, ndim).reshape(-1, ndim)
    test_y_rand = func(test_x_rand)
    test_x_fss = mlcs.sample_by_function(func, ndim=ndim, n_samples=n_test,
                                         minimize=False, range_in=(lboundx, uboundx),
                                         range_out=(lboundy, uboundy))
    test_y_fss = func(test_x_fss)

    mse_rand = np.zeros((n_reps, n_steps + 1))
    mse_fss = np.zeros((n_reps, n_steps + 1))

    plt.figure(figsize=(9, 6))

    for rep in range(n_reps):
        archive = mlcs.CandidateArchive(ndim=ndim, fidelities=['high'])
        archive.addcandidates(init_x, func(init_x))

        mfbo = mlcs.MultiFidelityBO(mff.forrester_sf, archive, output_range=(lboundy, uboundy), normalized=False)
        surr = mfbo.direct_models['high']

        mse_rand[rep, 0] = np.mean((test_y_rand - surr.predict(test_x_rand)) ** 2)
        mse_fss[rep, 0] = np.mean((test_y_fss - surr.predict(test_x_fss)) ** 2)

        for step in range(1, n_steps + 1):
            mfbo.iteration(iteration_idx=0)

            mse_rand[rep, step] = np.mean((test_y_rand - surr.predict(test_x_rand)) ** 2)
            mse_fss[rep, step] = np.mean((test_y_fss - surr.predict(test_x_fss)) ** 2)

        plt.plot(np.arange(n_steps + 1), mse_rand[rep], color=f'C{rep}', linestyle=':',
                 label=f'Random test set ({rep})')
        plt.plot(np.arange(n_steps + 1), mse_fss[rep], color=f'C{rep}', linestyle='-.', label=f'FSS test set ({rep})')
        plt.plot(np.arange(n_steps + 1), mse_rand[rep] - mse_fss[rep], color=f'C{rep}', label=f'Difference ({rep})')


    # Shrink current axis by 20%
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    #     plt.legend(loc=0)
    plt.axhline(y=0, color='black', linewidth=1)
    plt.xlim([0, n_steps])
    plt.title(f'Iterated pre-selected sample addition ({ndim}d)')
    plt.savefig(f'{plot_dir}rand_mse_over_presampling_{ndim}d.pdf')
    plt.savefig(f'{plot_dir}rand_mse_over_presampling_{ndim}d.png')
    plt.show()



if __name__ == '__main__':
    np.random.seed(20160501)
    for dim in [1, 2, 4, 8, 16]:
        iterative_EGO_sampling(ndim=dim)

