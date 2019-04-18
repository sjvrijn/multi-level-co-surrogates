# coding: utf-8


from pyDOE import lhs
import numpy as np
import os
import sys

from numba import jit, prange

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from itertools import product
import multifidelityfunctions as mff
import multiLevelCoSurrogates as mlcs

np.random.seed(20160501)  # Setting seed for reproducibility
OD = mff.oneDimensional


np.set_printoptions(linewidth=200)
plot_dir = '../multiLevelCoSurrogates/plots/'



min_high = 2
min_low = 3
max_high = 50
max_low = 125
num_reps = 50
step = 1 



def low_random_sample(ndim, nlow):
    return np.random.rand(nlow, ndim)

def low_lhs_sample(ndim, nlow):
    if ndim == 1:
        return np.linspace(0,1,nlow).reshape(-1,1)
    elif ndim > 1:
        return lhs(ndim, nlow)


#@jit(parallel=True)
def create_mse_tracking(func, sample_generator, 
                        max_high=40, max_low=100, num_reps=30,
                        min_high=2, min_low=3, step=1):
    ndim = func.ndim
    n_test_samples = 1000
    mse_tracking = np.empty((max_high+1, max_low+1, num_reps, 3))
    mse_tracking[:] = np.nan

    error_tracking = np.empty((max_high+1, max_low+1, num_reps, 3, n_test_samples))

    cases = list(product(range(min_high, max_high+1, step), range(min_low, max_low+1, step), range(num_reps)))


    input_range = mlcs.ValueRange(*np.array([func.l_bound, func.u_bound], dtype=np.float))
    output_range = (-10, 16)
    test_sample = mlcs.sample_by_function(func.high, n_samples=n_test_samples, ndim=func.ndim,
                                          range_in=input_range, range_out=output_range, minimize=False)

    np.save(f'{ndim}d_test_sample.npy', test_sample)

    #for idx, case in enumerate(cases):
    #    num_high, num_low, rep = case
    for num_high in range(min_high, max_high+1, step):
        for num_low in range(min_low, max_low+1, step):
            for rep in range(num_reps):
                idx = num_high * (max_high-min_high)/step \
                        + num_low * (max_low-min_low)/step \
                        + rep
                      

                if num_high >= num_low:
                    continue
                if idx % 100 == 0:
                    print(idx, '/', len(cases))

                low_x = sample_generator(ndim, num_low)
                high_x = low_x[np.random.choice(num_low, num_high, replace=False)]
        
                archive = mlcs.CandidateArchive(ndim=ndim, fidelities=['high', 'low', 'high-low'])
                archive.addcandidates(low_x, func.low(low_x), fidelity='low')
                archive.addcandidates(high_x, func.high(high_x), fidelity='high')

                mfbo = mlcs.MultiFidelityBO(func, archive, output_range=output_range, test_sample=test_sample)
                mse_tracking[num_high, num_low, rep] = mfbo.getMSE()
                for i, (direct, fid) in enumerate([(False, 'high'), (False, 'low'), (True, 'high')]):
                    if direct:
                        model = mfbo.direct_models
                    else:
                        model = mfbo.models

                    x = mfbo.mse_tester[fid].keywords['y_pred'].flatten()
                    y = model[fid].predict(mfbo.test_sample).flatten()
                    error_tracking[num_high, num_low, rep, i] = (x-y)**2

    print(len(cases), '/', len(cases))
    return mse_tracking, error_tracking


# ## Random Sample generation


inv_OD = mff.MultiFidelityFunction(
    u_bound=np.array(OD.u_bound), l_bound=np.array(OD.l_bound),
    functions=[lambda x: -OD.high(x), lambda x: -OD.low(x)],
    fidelity_names=['high', 'low'],
)


@mff.row_vectorize
def td_inv_high(xx):
    x1, x2 = xx
    return -(OD.high(x1) + OD.high(x2))

@mff.row_vectorize
def td_inv_low(xx):
    x1, x2 = xx
    return -(OD.low(x1) + OD.low(x2))


TD_inv = mff.MultiFidelityFunction(
    u_bound=np.array(OD.u_bound*2), l_bound=np.array(OD.l_bound*2),
    functions=[td_inv_high, td_inv_low],
    fidelity_names=['high', 'low'],
)

np.random.seed(20160501)  # Setting seed for reproducibility
mse_tracking, errors = create_mse_tracking(inv_OD, low_lhs_sample, 
                                           max_high=max_high, max_low=max_low, 
                                           num_reps=num_reps, min_high=min_high,
                                           min_low=min_low, step=step)

np.save('1d_lin_mse_tracking.npy', mse_tracking)
np.save('1d_lin_error_tracking.npy', errors)


np.random.seed(20160501)  # Setting seed for reproducibility
mse_tracking, errors = create_mse_tracking(TD_inv, low_lhs_sample, 
                                           max_high=max_high, max_low=max_low, 
                                           num_reps=num_reps, min_high=min_high, 
                                           min_low=min_low, step=step)

np.save('2d_lin_mse_tracking.npy', mse_tracking)
np.save('2d_lin_error_tracking.npy', errors)



