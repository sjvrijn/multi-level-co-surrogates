# coding: utf-8
import sys
import time
from collections import namedtuple
from functools import partial
from itertools import product
from multiprocessing import Pool, cpu_count

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
Instance = namedtuple('Instance', 'n_high n_low rep')


def low_lhs_sample(ndim, nlow):
    if ndim == 1:
        return np.linspace(0,1,nlow).reshape(-1,1)
    elif ndim > 1:
        return lhs(ndim, nlow)


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


def uniquify(sequence):
    """Reduce a list to it's unique elements while preserving order"""
    return list(dict.fromkeys(sequence))


def get_repr_surrogate_name(mfbo_options):
    """Create a representative name for the used surrogate according to
    `mfbo_options`. This is `surrogate_name` except when Kriging is used, then
    the kernel name is returned instead."""
    surr_name = mfbo_options.get('kernel') \
        if mfbo_options.get('surrogate_name', 'Kriging') == 'Kriging' \
        else mfbo_options['surrogate_name']
    return surr_name


def indexify(sequence, index_source):
    return [index_source.index(item) for item in sequence]


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


def extract_existing_instances(data):
    """Return a list of Instances that are non-NaN in the given xr.DataArray"""

    instance_coords = [data.coords[coord].values.tolist()
                       for coord in Instance._fields]

    array_instances = np.array([
        Instance(*instance)
        for instance in product(*instance_coords)
    ])

    indices = np.arange(np.prod(data.shape)).reshape(data.shape)
    valid_indices = np.where(np.isfinite(data), indices, np.nan).flatten()
    valid_indices = valid_indices[np.isfinite(valid_indices)].astype(int)

    return array_instances[valid_indices].tolist()


def filter_instances(instances, data):
    """Return `instances` with all instances removed that are already present in
    the file located at `output_path`"""

    existing_instances = extract_existing_instances(data)

    return [instance
            for instance in instances
            if instance not in existing_instances]


def create_hierarchical_model_instance(func, mfbo_options, ndim, instance):
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


def get_model_errors_for_instance(instance, func, mfbo_options, ndim):
    mfbo = create_hierarchical_model_instance(func, mfbo_options, ndim, instance)
    mses = mfbo.getMSE()
    r2s = mfbo.getR2()
    values = [model.predict(mfbo.test_sample).flatten()
              for model in [mfbo.models['high'],
                            mfbo.direct_models['high'],
                            mfbo.models['low']]
              ]
    return mses, r2s, values


def plot_model_and_samples(case, kernel, scaling_option, instance):
    """Create a multi-fidelity model based on given instance and show a plot of
    the surfaces and sampled points.
    Can be used for 1D or 2D functions."""
    options = {'kernel': kernel, 'scaling': scaling_option}
    mfbo = create_hierarchical_model_instance(case.func, options, case.ndim, instance)

    if case.ndim == 1:
        plot_x = np.linspace(case.func.l_bound, case.func.u_bound, 1001)

        plt.figure()
        plt.plot(plot_x, case.func.high(plot_x), label='True high-fidelity')
        plt.plot(plot_x, case.func.low(plot_x), label='True low-fidelity')
        plt.plot(plot_x, mfbo.models['high'].predict(plot_x), label='Hierarchical model')

        plt.scatter(*mfbo.archive.getcandidates(fidelity='high'), label='High-fidelity samples')
        plt.scatter(*mfbo.archive.getcandidates(fidelity='low'), label='low-fidelity samples')

        plt.title(f'{case.ndim}D {case.func.name}: {instance.n_high}/{instance.n_low}'
                  f' samples (repetition {instance.rep})')
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

    elif case.ndim == 2:
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

    else:
        raise ValueError(f"Dimensionality case.ndim={case.ndim} not supported by"
                         f"plot_model_and_samples. Only 1D and 2D are supported")


def create_mse_tracking(func, ndim, mfbo_options, instances):

    n_test_samples = mfbo_options['test_sample'].shape[0]
    models = ['high_hier', 'high', 'low']

    indices = indexify_instances(instances)
    n_highs, n_lows, reps = map(uniquify, zip(*instances))
    array_size = (len(n_highs), len(n_lows), len(reps), len(models))

    mse_tracking = np.full(array_size, np.nan)
    r2_tracking = np.full(array_size, np.nan)
    value_tracking = np.full((*array_size, n_test_samples), np.nan)

    # if mfbo_options.get('surrogate_name', 'Kriging') == 'Kriging':

    print('starting loops')
    for i, (instance, index) in enumerate(zip(instances, indices)):
        if i % 100 == 0:
            print(f'{i}/{len(instances)}')

        mses, r2s, values = get_model_errors_for_instance(instance, func,
                                                          mfbo_options, ndim)
        mse_tracking[index] = mses
        r2_tracking[index] = r2s
        value_tracking[index] = values


    print(f'{len(instances)}/{len(instances)}')

    # else:
    #     f = partial(get_model_errors_for_instance,
    #                 func=func,
    #                 mfbo_options=mfbo_options,
    #                 ndim=ndim)
    #
    #     with Pool(cpu_count()) as p:
    #         all_results = p.map(f, instances)
    #
    #     for index, (mses, r2s, values) in zip(indices, zip(*all_results)):
    #         mse_tracking[index] = mses
    #         r2_tracking[index] = r2s
    #         value_tracking[index] = values


    # Iteration finished, arranging data into xr.Dataset
    attributes = dict(experiment='create_mse_tracking',
                      function=func.name, ndim=ndim,
                      kernel=mfbo_options.get('kernel', 'N/A'),
                      surrogate_name=mfbo_options.get('surrogate_name', 'Kriging'),
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


def create_model_error_grid(case, mfbo_options, instances, save_dir):
    """Create a grid of model errors for the given MFF-function case at the
    given list of instances.
    The results are saved in a NETCDF .nc file at the specified `save_dir`"""

    start = time.time()
    print(f"Starting case {case} at {start}")

    #TODO: make use of dict.get(key, default) instead
    surr_name = get_repr_surrogate_name(mfbo_options)

    output_path = save_dir / f"{surr_name}-{case.ndim}d-{case.func.name}.nc"
    if output_path.exists():
        with xr.open_dataset(output_path) as ds:
            da = ds['mses'].load()
            instances = filter_instances(instances, da.sel(model='high_hier'))

    test_sample = get_test_sample(case.ndim, save_dir)
    mfbo_options['test_sample'] = test_sample
    output = create_mse_tracking(func=case.func, mfbo_options=mfbo_options,
                                 ndim=case.ndim, instances=instances)
    if output_path.exists():
        with xr.load_dataset(output_path) as ds:
            output = ds.merge(output)
    output.to_netcdf(output_path)

    end = time.time()
    print(f"Ended case {case} at {end}\n"
          f"Time spent: {end - start}")
