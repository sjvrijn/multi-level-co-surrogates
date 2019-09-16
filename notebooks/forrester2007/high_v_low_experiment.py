# coding: utf-8
from itertools import product
from functools import partial

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import sys


module_path = Path().joinpath('../..')
if module_path not in sys.path:
    sys.path.append(module_path)

import multiLevelCoSurrogates as mlcs
from function_defs import low_lhs_sample, red_dot, blue_circle

np.random.seed(20160501)  # Setting seed for reproducibility

np.set_printoptions(linewidth=200)
plot_dir = Path('../../plots/')
file_dir = Path('../../files/')
mlcs.guaranteeFolderExists(plot_dir)
mlcs.guaranteeFolderExists(file_dir)


def uniquify(sequence):
    """Reduce a list to it's unique elements while preserving order"""
    return list(dict.fromkeys(sequence))


def indexify(sequence, index_source):
    return [index_source.index(item) for item in sequence]


def create_mse_tracking(func, ndim, mfbo_options, instances):

    n_test_samples = 500*ndim

    models = ['high_hier', 'high', 'low']
    n_highs, n_lows, reps = map(uniquify, zip(*instances))
    array_size = (len(n_highs), len(n_lows), len(reps), 3)

    mse_tracking = np.full(array_size, np.nan)
    r2_tracking = np.full(array_size, np.nan)
    value_tracking = np.full((*array_size, n_test_samples), np.nan)

    test_sample = low_lhs_sample(ndim, n_test_samples)  #TODO: consider rescaling test_sample here instead of in MultiFidBO
    np.save(file_dir.joinpath(f'{ndim}d_test_sample.npy'), test_sample)
    mfbo_options['test_sample'] = test_sample

    print('starting loops')

    #TODO: simplify this index creation stuff
    all_n_high, all_n_low, all_reps = list(zip(*instances))

    n_high_indices = indexify(all_n_high, n_highs)
    n_low_indices = indexify(all_n_low, n_lows)
    reps_indices = indexify(all_reps, reps)

    indices = list(zip(n_high_indices, n_low_indices, reps_indices))

    for i, (instance, indices) in enumerate(zip(instances, indices)):

        if i % 100 == 0:
            print(f'{i}/{len(instances)}')

        mfbo = create_experiment_instance(func, mfbo_options, ndim, instance)

        mse_tracking[indices] = mfbo.getMSE()
        r2_tracking[indices] = mfbo.getR2()

        for m, model in enumerate([mfbo.models['high'],
                                   mfbo.direct_models['high'],
                                   mfbo.models['low']]):
            value_tracking[(*indices), m] = model.predict(mfbo.test_sample).flatten()

    print(f'{len(instances)}/{len(instances)}')

    attributes = dict(experiment='create_mse_tracking',
                      function=func.name, ndim=ndim,
                      kernel=mfbo_options['kernel'],
                      scaling=mfbo_options['scaling'])

    mse_tracking = xr.DataArray(mse_tracking,
                                dims=['n_high', 'n_low', 'rep', 'model'],
                                coords=[n_highs, n_lows, reps, models],
                                attrs=attributes)
    r2_tracking = xr.DataArray(r2_tracking,
                                dims=['n_high', 'n_low', 'rep', 'model'],
                                coords=[n_highs, n_lows, reps, models],
                                attrs=attributes)
    value_tracking = xr.DataArray(value_tracking,
                                  dims=['n_high', 'n_low', 'rep', 'model', 'idx'],
                                  coords={'n_high': n_highs, 'n_low': n_lows,
                                          'rep': reps, 'model': models,
                                          'idx': range(n_test_samples)},
                                  attrs=attributes)

    output = xr.Dataset({'mses': mse_tracking,
                         'r2': r2_tracking,
                         'values': value_tracking})

    return output


def create_experiment_instance(func, mfbo_options, ndim, instance):
    """Create a consistent instantiated MFBO instance with the given parameters.

    :returns MultiFidelityBO instance
    """
    num_high, num_low, rep = instance

    np.random.seed(int(f'{num_high:03}{num_low:03}{rep:03}'))

    high_x, low_x = multi_fidelity_doe(ndim, num_high, num_low)

    range_out = (np.array(func.l_bound), np.array(func.u_bound))

    high_x = mlcs.rescale(high_x, range_in=(0,1), range_out=range_out)
    low_x = mlcs.rescale(low_x, range_in=(0,1), range_out=range_out)

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

        output = create_mse_tracking(func=case.func, mfbo_options=options,
                                     ndim=case.ndim, instances=instances)

        base_file_name = f'{k}{case.ndim}d_{case.func.name}'

        np.save(file_dir.joinpath(f'{base_file_name}_instances.npy'), np.array(instances))
        output.to_netcdf(file_dir.joinpath(f'{base_file_name}.nc'))


def plot_model_and_samples(case, kernel, scaling_option, instance):
    options = {'kernel': kernel[:-1], 'scaling': scaling_option}
    mfbo = create_experiment_instance(case.func, options, case.ndim, instance)

    if case.ndim == 1:

        plot_x = np.linspace(case.func.l_bound, case.func.u_bound, 1001)

        plt.figure()
        plt.plot(plot_x, case.func.high(plot_x), label='True high-fidelity')
        plt.plot(plot_x, case.func.low(plot_x), label='True low-fidelity')
        plt.plot(plot_x, mfbo.models['high'].predict(plot_x), label='Hierarchical model')

        plt.scatter(*mfbo.archive.getcandidates(fidelity='high'), label='High-fidelity samples')
        plt.scatter(*mfbo.archive.getcandidates(fidelity='low'), label='low-fidelity samples')

        plt.title(f'{case.ndim}D {case.func.name}: {instance.high}/{instance.low}'
                  f' samples (repetition {instance.rep})')
        plt.legend(loc=0)
        plt.tight_layout()

        plt.show()

    else:

        surf_high = mlcs.createsurface(case.func.high, l_bound=case.func.l_bound, u_bound=case.func.u_bound)
        surf_low = mlcs.createsurface(case.func.low, l_bound=case.func.l_bound, u_bound=case.func.u_bound)

        surf_model_high = mlcs.createsurface(partial(mlcs.gpplot, func=mfbo.models['high'].predict),
                                             l_bound=case.func.l_bound,
                                             u_bound=case.func.u_bound)
        surf_model_low = mlcs.createsurface(partial(mlcs.gpplot, func=mfbo.models['low'].predict),
                                             l_bound=case.func.l_bound,
                                             u_bound=case.func.u_bound)

        points = [mlcs.ScatterPoints(*mfbo.archive.getcandidates(fidelity='high'), red_dot),
                  mlcs.ScatterPoints(*mfbo.archive.getcandidates(fidelity='low'), blue_circle)]

        mlcs.plotsurfaces([surf_high, surf_low, surf_model_high, surf_model_low],
                          titles=['True High', 'True Low', 'Hierarchical model', 'Low-fidelity model'],
                          all_points=[points, points, points, points], shape=(2,2))
