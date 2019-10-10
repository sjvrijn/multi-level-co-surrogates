# coding: utf-8
import sys
import time
from collections import namedtuple
from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pyDOE import lhs
from pyprojroot import here
from scipy.spatial import distance

module_path = str(here())
if module_path not in sys.path:
    sys.path.append(module_path)

import multiLevelCoSurrogates as mlcs


# defining some point styles
red_dot = {'marker': '.', 'color': 'red'}
blue_circle = {'marker': 'o', 'facecolors': 'none', 'color': 'blue'}


Case = namedtuple('Case', 'ndim func')
Instance = namedtuple('Instance', 'high low rep')


def low_lhs_sample(ndim, nlow):
    if ndim == 1:
        return np.linspace(0,1,nlow).reshape(-1,1)
    elif ndim > 1:
        return lhs(ndim, nlow)


def uniquify(sequence):
    """Reduce a list to it's unique elements while preserving order"""
    return list(dict.fromkeys(sequence))


def indexify(sequence, index_source):
    return [index_source.index(item) for item in sequence]


def extract_existing_instances(data):
    """Return a list of Instances that are non-NaN in the given xr.DataArray"""

    all_instances = product(*[x.values.tolist() for x in data.coords.values()])
    all_instances = list(map(Instance, all_instances))
    array_instances = np.array(all_instances)

    indices = np.arange(np.prod(data.shape)).reshape(data.shape)
    valid_indices = np.where(np.isfinite(data), indices, np.nan).flatten()
    valid_indices = valid_indices[np.isfinite(valid_indices)].astype(int)

    return array_instances[valid_indices].tolist()


def extend_existing_dataset(dataset, instances):
    """Take the given dataset, extend it with all given instances not already present
    in the dataset, and return this extended version."""
    pass


def create_mse_tracking(func, ndim, mfbo_options, instances):

    n_test_samples = mfbo_options['test_sample'].shape[0]
    models = ['high_hier', 'high', 'low']

    n_highs, n_lows, reps = map(uniquify, zip(*instances))
    array_size = (len(n_highs), len(n_lows), len(reps), len(models))

    mse_tracking = np.full(array_size, np.nan)
    r2_tracking = np.full(array_size, np.nan)
    value_tracking = np.full((*array_size, n_test_samples), np.nan)

    indices = indexify_instances(instances)

    print('starting loops')
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

    # Iteration finished, arranging data into xr.Dataset
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


def indexify_instances(instances):
    """Return an 'indexified' version of the list of instances, i.e. each
    instance is replaced by a tuple of indices.
    These indices can then be used to write values into a size-correct numpy-
    array as part of a xr.DataArray whose coordinates do not necessarily
    start at 0 or increase by 1.
    """
    n_highs, n_lows, reps = map(uniquify, zip(*instances))

    all_n_high, all_n_low, all_reps = list(zip(*instances))
    n_high_indices = indexify(all_n_high, n_highs)
    n_low_indices = indexify(all_n_low, n_lows)
    reps_indices = indexify(all_reps, reps)
    indices = list(zip(n_high_indices, n_low_indices, reps_indices))
    return indices


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
    high_x = low_lhs_sample(ndim, num_high)
    low_x = low_lhs_sample(ndim, num_low)

    dists = distance.cdist(high_x, low_x)

    #TODO: this is the naive method, potentially speed up?
    highs_to_match = set(range(num_high))
    while highs_to_match:
        min_dist = np.min(dists)
        high_idx, low_idx = np.argwhere(dists == min_dist)[0]

        low_x[low_idx] = high_x[high_idx]

        dists[high_idx,:] = np.inf
        dists[:,low_idx] = np.inf
        highs_to_match.remove(high_idx)
    return high_x, low_x


def filter_instances(instances, data):
    """Return `instances` with all instances removed that are already present in
    the file located at `output_path`"""

    data.load()
    existing_instances = extract_existing_instances(data)

    return [instance
            for instance in instances
            if instance not in existing_instances]


def calculate_mse_grid(cases, kernels, scaling_options, instances, save_dir):

    for case, k, scale in product(cases, kernels, scaling_options):
        start = time.time()
        print(f"Starting case {case} at {start}")

        output_path = save_dir / f"{k}{case.ndim}d-{case.func.name}.nc"
        if output_path.exists():
            ds = xr.open_dataset(output_path)
            instances = filter_instances(instances, ds['mses'])
        else:
            ds = None

        test_sample = get_test_sample(case.ndim, save_dir)
        options = {'kernel': k[:-1], 'scaling': scale, 'test_sample': test_sample}
        output = create_mse_tracking(func=case.func, mfbo_options=options,
                                     ndim=case.ndim, instances=instances)

        if ds:
            ds.load()
            output = ds.merge(output)

        output.to_netcdf(output_path)
        end = time.time()
        print(f"Ended case {case} at {end}\n"
              f"Time spent: {end-start}")


def get_test_sample(ndim, save_dir):
    """Get the test-sample for an `ndim`-dimensional function. If a sample has
    been previously generated and saved, load it from file. Else, generate it
    based on the fixed seed and store it for future use."""
    test_sample_save_name = save_dir / f'{ndim}d-test-sample.npy'
    if test_sample_save_name.exists():
        return np.load(test_sample_save_name)

    n_test_samples = 500 * ndim
    np.random.seed(20160501)  # Setting seed for reproducibility
    test_sample = low_lhs_sample(ndim, n_test_samples)
    np.save(test_sample_save_name, test_sample)
    return test_sample


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
        bounds = {'l_bound': case.func.l_bound, 'u_bound': case.func.u_bound}

        surf_high = mlcs.createsurface(case.func.high, **bounds)
        surf_low = mlcs.createsurface(case.func.low, **bounds)

        surf_model_high = mlcs.createsurface(partial(mlcs.gpplot, func=mfbo.models['high'].predict),
                                             **bounds)
        surf_model_low = mlcs.createsurface(partial(mlcs.gpplot, func=mfbo.models['low'].predict),
                                             **bounds)

        points = [mlcs.ScatterPoints(*mfbo.archive.getcandidates(fidelity='high'), red_dot),
                  mlcs.ScatterPoints(*mfbo.archive.getcandidates(fidelity='low'), blue_circle)]

        mlcs.plotsurfaces([surf_high, surf_low, surf_model_high, surf_model_low],
                          titles=['True High', 'True Low', 'Hierarchical model', 'Low-fidelity model'],
                          all_points=[points, points, points, points], shape=(2,2))
