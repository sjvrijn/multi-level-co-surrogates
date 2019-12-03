# coding: utf-8
import sys
import time
from collections import namedtuple
from functools import partial
from itertools import product
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


Instance = namedtuple('Instance', 'n_high n_low rep')
BiFidelityDoE = namedtuple("BiFidelityDoE", "high low")

def set_seed_by_instance(num_high, num_low, rep):
    """Fix the numpy random seed based on an Instance"""
    np.random.seed(int(f'{num_high:03}{num_low:03}{rep:03}'))


def low_lhs_sample(ndim, nlow):
    if ndim == 1:
        return np.linspace(0,1,nlow).reshape(-1,1)
    elif ndim > 1:
        return lhs(ndim, nlow)


def bi_fidelity_doe(ndim, num_high, num_low):
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
    return BiFidelityDoE(high_x, low_x)


def subselect_bi_fidelity_doe(DoE, num_high, num_low):
    """Given an existing bi-fidelity Design of Experiments (DoE) `high, low`,
    creates a subselection of given size `num_high, num_low` based on uniform
    selection. The subselection maintains the property that all high-fidelity
    samples are a subset of the low-fidelity samples.

    Raises a `ValueError` if invalid `num_high` or `num_low` are given."""
    high, low = DoE
    if not 1 < num_high < len(high):
        raise ValueError(f"'num_high' must be in the range [2, len(DoE.high)], but is {num_high}")
    elif num_low > len(low):
        raise ValueError(f"'num_low' cannot be greater than len(DoE.low), but is {num_low}")
    elif num_low <= num_high:
        raise ValueError(f"'num_low' must be greater than 'num_high', but {num_low} <= {num_high}")

    sub_high = high[np.random.choice(len(high), num_high, replace=False)]

    if num_low == len(low):
        sub_low = low
    else:
        # remove all sub_high from low
        filtered_low = np.array([x for x in low if x not in sub_high])
        # randomly select (num_low - num_high) remaining
        extra_low = filtered_low[
            np.random.choice(len(filtered_low), num_low - num_high, replace=False)
        ]
        # concatenate sub_high with selected sub_low
        sub_low = np.concatenate([sub_high, extra_low], axis=0)

    return BiFidelityDoE(sub_high, sub_low)


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


def repr_surrogate_name(mfbo_options):
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


def scale_to_function(func, xx, range_in=mlcs.ValueRange(0, 1)):
    range_out = (np.array(func.l_bound), np.array(func.u_bound))
    return [mlcs.rescale(x, range_in=range_in, range_out=range_out) for x in xx]


def plot_model_and_samples(func, kernel, scaling_option, instance):
    """Create a multi-fidelity model based on given instance and show a plot of
    the surfaces and sampled points.
    Can be used for 1D or 2D functions."""
    options = {'kernel': kernel, 'scaling': scaling_option}

    num_high, num_low, rep = instance
    set_seed_by_instance(num_high, num_low, rep)



    high_x, low_x = bi_fidelity_doe(func.ndim, num_high, num_low)
    high_x, low_x = scale_to_function(func, [high_x, low_x])
    high_y, low_y = func.high(high_x), \
                    func.low(low_x)

    archive = mlcs.CandidateArchive.from_multi_fidelity_function(func, ndim=func.ndim)
    archive.addcandidates(low_x, low_y, fidelity='low')
    archive.addcandidates(high_x, high_y, fidelity='high')

    mfbo = mlcs.MultiFidelityBO(func, archive, **options)



    if func.ndim == 1:
        plot_x = np.linspace(func.l_bound, func.u_bound, 1001)

        plt.figure()
        plt.plot(plot_x, func.high(plot_x), label='True high-fidelity')
        plt.plot(plot_x, func.low(plot_x), label='True low-fidelity')
        plt.plot(plot_x, mfbo.models['high'].predict(plot_x), label='Hierarchical model')

        plt.scatter(*mfbo.archive.getcandidates(fidelity='high'),
                    label='High-fidelity samples')
        plt.scatter(*mfbo.archive.getcandidates(fidelity='low'),
                    label='low-fidelity samples')

        plt.title(f'{func.ndim}D {func.name}: {instance.n_high}/{instance.n_low}'
                  f' samples (repetition {instance.rep})')
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

    elif func.ndim == 2:
        bounds = {'l_bound': func.l_bound, 'u_bound': func.u_bound}

        surf_high = mlcs.createsurface(func.high, **bounds)
        surf_low = mlcs.createsurface(func.low, **bounds)

        surf_model_high = mlcs.createsurface(partial(mlcs.gpplot,
                                                     func=mfbo.models['high'].predict),
                                             **bounds)
        surf_model_low = mlcs.createsurface(partial(mlcs.gpplot,
                                                    func=mfbo.models['low'].predict),
                                             **bounds)

        points = [mlcs.ScatterPoints(*mfbo.archive.getcandidates(fidelity='high'),
                                     red_dot),
                  mlcs.ScatterPoints(*mfbo.archive.getcandidates(fidelity='low'),
                                     blue_circle)]

        mlcs.plotsurfaces([surf_high, surf_low, surf_model_high, surf_model_low],
                          titles=['True High', 'True Low',
                                  'Hierarchical model', 'Low-fidelity model'],
                          all_points=[points, points, points, points], shape=(2,2))

    else:
        raise ValueError(f"Dimensionality case.func.ndim={func.ndim} not supported by"
                         f"plot_model_and_samples. Only 1D and 2D are supported")


def create_model_error_grid(func, instances, mfbo_options, save_dir):
    """Create a grid of model errors for the given MFF-function case at the
    given list of instances.
    The results are saved in a NETCDF .nc file at the specified `save_dir`"""

    start = time.time()
    print(f"Starting case {func} at {start}")

    # Determine unique output path for this experiment
    surr_name = repr_surrogate_name(mfbo_options)
    output_path = save_dir / f"{surr_name}-{func.ndim}d-{func.name}.nc"


    # Don't redo any prior data that already exists
    if output_path.exists():
        with xr.open_dataset(output_path) as ds:
            da = ds['mses'].load()
            instances = filter_instances(instances, da.sel(model='high_hier'))

        # Return early if there is nothing left to do
        if not instances:
            return


    # Setup some (final) options for the hierarchical model
    mfbo_options['test_sample'] = get_test_sample(func.ndim, save_dir)


    results = []
    print('starting loops')
    for i, (num_high, num_low, rep) in enumerate(instances):
        if i % 100 == 0:
            print(f'{i}/{len(instances)}')

        set_seed_by_instance(num_high, num_low, rep)

        # Create Multi-Fidelity DoE in- and output according to instance specification
        high_x, low_x = bi_fidelity_doe(func.ndim, num_high, num_low)
        high_x, low_x = scale_to_function(func, [high_x, low_x])
        high_y, low_y = func.high(high_x), \
                        func.low(low_x)

        # Create an archive from the MF-function and MF-DoE data
        archive = mlcs.CandidateArchive.from_multi_fidelity_function(func, ndim=func.ndim)
        archive.addcandidates(low_x, low_y, fidelity='low')
        archive.addcandidates(high_x, high_y, fidelity='high')

        # (Automatically) Create the hierarchical model
        mfbo = mlcs.MultiFidelityBO(func, archive, **mfbo_options)

        # Get the results we're interested in from the model for this instance
        mses = mfbo.getMSE()
        r2s = mfbo.getR2()
        values = [model.predict(mfbo.test_sample).flatten()
                  for model in [mfbo.models['high'],
                                mfbo.direct_models['high'],
                                mfbo.models['low']]
                  ]

        # Store the results
        results.append((mses, r2s, values))

    print(f'{len(instances)}/{len(instances)}')



    # Create attributes dictionary
    attributes = dict(experiment='create_model_error_grid',
                      function=func.name,
                      ndim=func.ndim,
                      kernel=mfbo_options.get('kernel', 'N/A'),
                      surrogate_name=mfbo_options.get('surrogate_name', 'Kriging'),
                      scaling=mfbo_options['scaling'])

    ## Iteration finished, arranging data into xr.Dataset
    output = results_to_dataset(results, instances, mfbo_options, attributes)


    # Merge with prior existing data
    # NOTE: even if `output` is empty, attributes will be overwitten/updated
    if output_path.exists():
        with xr.load_dataset(output_path) as ds:
            output = output.merge(ds)

    # Store results
    output.to_netcdf(output_path)

    end = time.time()
    print(f"Ended case {func} at {end}\n"
          f"Time spent: {end - start}")


def create_resampling_error_grid(func, DoE_spec, instances, mfbo_options, save_dir):
    """Create a grid of model errors for the given MFF-function at the
    given list of instances, with all data for training the model being based
    on an initial given DoE specification.
    The results are saved in a NETCDF .nc file at the specified `save_dir`"""

    start = time.time()
    print(f"Starting case {func} at {start}")

    # Determine unique output path for this experiment
    surr_name = repr_surrogate_name(mfbo_options)
    doe_high, doe_low = DoE_spec
    output_path = save_dir / f"{surr_name}-{func.ndim}d-{func.name}-sub{doe_high}-{doe_low}.nc"


    # Don't redo any prior data that already exists
    if output_path.exists():
        with xr.open_dataset(output_path) as ds:
            da = ds['mses'].load()
            instances = filter_instances(instances, da.sel(model='high_hier'))

        # Return early if there is nothing left to do
        if not instances:
            return


    # Setup some (final) options for the hierarchical model
    mfbo_options['test_sample'] = get_test_sample(func.ndim, save_dir)


    # Create initial DoE
    np.random.seed(20160501)  # Setting seed for reproducibility
    DoE = bi_fidelity_doe(func.ndim, doe_high, doe_low)
    DoE = scale_to_function(func, DoE)


    results = []
    print('starting loops')
    for i, (num_high, num_low, rep) in enumerate(instances):
        if i % 100 == 0:
            print(f'{i}/{len(instances)}')

        set_seed_by_instance(num_high, num_low, rep)

        # Create sub-sampled Multi-Fidelity DoE in- and output according to instance specification
        high_x, low_x = subselect_bi_fidelity_doe(DoE, num_high, num_low)
        # TODO: precompute output values and include them when subselecting
        high_y, low_y = func.high(high_x), \
                        func.low(low_x)

        # Create an archive from the MF-function and MF-DoE data
        archive = mlcs.CandidateArchive.from_multi_fidelity_function(func, ndim=func.ndim)
        archive.addcandidates(low_x, low_y, fidelity='low')
        archive.addcandidates(high_x, high_y, fidelity='high')

        # (Automatically) Create the hierarchical model
        mfbo = mlcs.MultiFidelityBO(func, archive, **mfbo_options)

        # Get the results we're interested in from the model for this instance
        mses = mfbo.getMSE()
        r2s = mfbo.getR2()
        values = [model.predict(mfbo.test_sample).flatten()
                  for model in [mfbo.models['high'],
                                mfbo.direct_models['high'],
                                mfbo.models['low']]
                  ]

        # Store the results
        results.append((mses, r2s, values))

    print(f'{len(instances)}/{len(instances)}')



    # Create attributes dictionary
    attributes = dict(experiment='create_resampling_error_grid',
                      doe=f"{doe_high},{doe_low}",
                      function=func.name,
                      ndim=func.ndim,
                      kernel=mfbo_options.get('kernel', 'N/A'),
                      surrogate_name=mfbo_options.get('surrogate_name', 'Kriging'),
                      scaling=mfbo_options['scaling'])

    ## Iteration finished, arranging data into xr.Dataset
    output = results_to_dataset(results, instances, mfbo_options, attributes)


    # Merge with prior existing data
    # NOTE: even if `output` is empty, attributes will be overwitten/updated
    if output_path.exists():
        with xr.load_dataset(output_path) as ds:
            output = output.merge(ds)

    # Store results
    output.to_netcdf(output_path)

    end = time.time()
    print(f"Ended case {func} at {end}\n"
          f"Time spent: {end - start}")


def results_to_dataset(results, instances, mfbo_options, attributes):
    """"Manually creating numpy arrays to store the data for eventual
    reading in as XArray DataArray/DataSet"""

    n_test_samples = mfbo_options['test_sample'].shape[0]

    # Hardcoded model names
    models = ['high_hier', 'high', 'low']

    # Get lists of unique entries per instance parameter
    n_highs, n_lows, reps = map(uniquify, zip(*instances))

    # Create empty numpy arrays with the correct minimal size according
    common_shape = (len(n_highs), len(n_lows), len(reps), len(models))
    mse_tracking = np.full(common_shape, np.nan)
    r2_tracking = np.full(common_shape, np.nan)
    value_tracking = np.full((*common_shape, n_test_samples), np.nan)

    # Create explicit indices at which values have to be inserted in numpy arrays
    indices = indexify_instances(instances)

    # Transfer the results from sipmle list to numpy array
    for index, (mses, r2s, values) in zip(indices, results):
        mse_tracking[index] = mses
        r2_tracking[index] = r2s
        value_tracking[index] = values

    # Create separate DataArrays for each numpy array
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


    ### Pandas-based alternative:
    # mses, r2s, values = [np.array(x) for x in zip(*results)]
    #
    # multi_index_values = [(*instance, model) for instance, model in
    #                       product(instances, models)]
    # multi_index = pd.MultiIndex.from_tuples(tuples=multi_index_values,
    #                                         names=['n_high', 'n_low', 'rep', 'model'])
    # mse_tracking = xr.DataArray.from_series(pd.Series(data=mses.flatten(), index=multi_index))
    # r2_tracking = xr.DataArray.from_series(pd.Series(data=r2s.flatten(), index=multi_index))
    #
    # multi_index_values = [(*instance, model, idx) for instance, model, idx in
    #                       product(instances, models, range(n_test_samples))]
    # multi_index = pd.MultiIndex.from_tuples(tuples=multi_index_values,
    #                                         names=['n_high', 'n_low', 'rep', 'model', 'idx'])
    # value_tracking = xr.DataArray.from_series(pd.Series(data=values.flatten(), index=multi_index))


    # Assembling Dataset from DataArrays
    output = xr.Dataset({'mses': mse_tracking,
                         'r2': r2_tracking,
                         'values': value_tracking})

    return output
