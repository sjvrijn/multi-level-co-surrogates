# coding: utf-8
import sys
from datetime import datetime
from collections import namedtuple
from functools import partial
from itertools import product
from pathlib import Path
from typing import Sequence, List, Dict, Tuple, Any, NamedTuple
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from parse import parse
from pyprojroot import here
from scipy.spatial import distance
from sklearn.metrics import mean_squared_error, r2_score
from mf2 import MultiFidelityFunction

module_path = str(here())
if module_path not in sys.path:
    sys.path.append(module_path)

import multiLevelCoSurrogates as mlcs


# defining some point styles
red_dot = {'marker': '.', 'color': 'red'}
blue_circle = {'marker': 'o', 'facecolors': 'none', 'color': 'blue'}


Instance = namedtuple('Instance', 'n_high n_low rep')
BiFidelityDoE = namedtuple("BiFidelityDoE", "high low")

def set_seed_by_instance(num_high: int, num_low: int, rep: int) -> None:
    """Fix the numpy random seed based on an Instance"""
    np.random.seed(int(f'{num_high:03}{num_low:03}{rep:03}'))


def bi_fidelity_doe(ndim: int, num_high: int, num_low: int) -> BiFidelityDoE:
    """Create a Design of Experiments (DoE) for two fidelities in `ndim`
    dimensions. The high-fidelity samples are guaranteed to be a subset
    of the low-fidelity samples.

    :returns high-fidelity samples, low-fidelity samples
    """
    high_x = mlcs.low_lhs_sample(ndim, num_high)
    low_x = mlcs.low_lhs_sample(ndim, num_low)

    dists = distance.cdist(high_x, low_x)

    #TODO: this is the naive method, potentially speed up?
    highs_to_match = set(range(num_high))
    while highs_to_match:
        min_dist = np.min(dists)
        high_idx, low_idx = np.argwhere(dists == min_dist)[0]

        low_x[low_idx] = high_x[high_idx]
        # make sure just selected samples are not re-selectable
        dists[high_idx,:] = np.inf
        dists[:,low_idx] = np.inf
        highs_to_match.remove(high_idx)
    return BiFidelityDoE(high_x, low_x)


def split_bi_fidelity_doe(DoE: BiFidelityDoE, num_high: int, num_low: int) -> Tuple[BiFidelityDoE, BiFidelityDoE]:
    """Given an existing bi-fidelity Design of Experiments (DoE) `high, low`,
    creates a subselection of given size `num_high, num_low` based on uniform
    selection. The subselection maintains the property that all high-fidelity
    samples are a subset of the low-fidelity samples.

    Raises a `ValueError` if invalid `num_high` or `num_low` are given."""
    high, low = DoE
    if not 1 < num_high < len(high):
        raise ValueError(f"'num_high' must be in the range [2, len(DoE.high) (={len(DoE.high)})], but is {num_high}")
    elif num_low > len(low):
        raise ValueError(f"'num_low' cannot be greater than len(DoE.low) (={len(DoE.low)}), but is {num_low}")
    elif num_low <= num_high:
        raise ValueError(f"'num_low' must be greater than 'num_high', but {num_low} <= {num_high}")

    indices = np.random.permutation(len(high))
    sub_high, leave_out_high = high[indices[:num_high]], high[indices[num_high:]]

    if num_low == len(low):
        sub_low = low
        leave_out_low = []
    else:
        # remove all sub_high from low
        filtered_low = np.array([x for x in low if x not in sub_high])
        # randomly select (num_low - num_high) remaining
        indices = np.random.permutation(len(filtered_low))
        num_low_left = num_low - num_high
        extra_low, leave_out_low = filtered_low[indices[:num_low_left]], \
                                   filtered_low[indices[num_low_left:]]
        # concatenate sub_high with selected sub_low
        sub_low = np.concatenate([sub_high, extra_low], axis=0)

    selected = BiFidelityDoE(sub_high, sub_low)
    left_out = BiFidelityDoE(leave_out_high, leave_out_low)
    return selected, left_out


def get_test_sample(ndim: int, save_dir: Path) -> np.ndarray:
    """Get the test-sample for an `ndim`-dimensional function. If a sample has
    been previously generated and saved, load it from file. Else, generate it
    based on the fixed seed and store it for future use."""
    test_sample_save_name = save_dir / f'{ndim}d-test-sample.npy'
    if test_sample_save_name.exists():
        return np.load(test_sample_save_name)

    n_test_samples = 500 * ndim
    np.random.seed(20160501)  # Setting seed for reproducibility
    test_sample = mlcs.low_lhs_sample(ndim, n_test_samples)
    np.save(test_sample_save_name, test_sample)
    return test_sample


def uniquify(sequence: Sequence) -> List:
    """Reduce a list to it's unique elements while preserving order"""
    return list(dict.fromkeys(sequence))


def repr_surrogate_name(mfbo_options: Dict[str, Any]) -> str:
    """Create a representative name for the used surrogate according to
    `mfbo_options`. This is `surrogate_name` except when Kriging is used, then
    the kernel name is returned instead."""
    surr_name = mfbo_options.get('kernel') \
        if mfbo_options.get('surrogate_name', 'Kriging') == 'Kriging' \
        else mfbo_options['surrogate_name']
    return surr_name


def indexify(sequence: Sequence, index_source: Sequence) -> List:
    """For the given `sequence`, get the index of each item in the `index_source`"""
    return [index_source.index(item) for item in sequence]


def indexify_instances(instances: Sequence[Instance]) -> List:
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


def extract_existing_instances(data: xr.DataArray) -> List:
    """Return a list of Instances that are non-NaN in the given xr.DataArray"""
    instance_coords = [
        data.coords[coord].values.tolist()
        for coord in Instance._fields
    ]

    array_instances = np.array([
        Instance(*instance)
        for instance in product(*instance_coords)
    ])

    indices = np.arange(np.prod(data.shape)).reshape(data.shape)
    valid_indices = np.where(np.isfinite(data), indices, np.nan).flatten()
    valid_indices = valid_indices[np.isfinite(valid_indices)].astype(int)

    return array_instances[valid_indices].tolist()


def filter_instances(instances: Sequence[Instance], data: xr.DataArray) -> List[Instance]:
    """Return `instances` with all instances removed that are already present in
    the file located at `output_path`"""

    existing_instances = extract_existing_instances(data)
    existing_instances = set(map(tuple, existing_instances))

    return [Instance(*instance)
            for instance in map(tuple, instances)
            if instance not in existing_instances]


def scale_to_function(func: MultiFidelityFunction, xx: Sequence[Sequence], range_in=mlcs.ValueRange(0, 1)) -> List[Sequence]:
    """Scale the input data `xx` from `range_in` to the bounds of the given function.
    :param range_in: defined range from which input values were drawn. Default: (0,1)
    """
    range_out = (np.array(func.l_bound), np.array(func.u_bound))
    return [mlcs.rescale(x, range_in=range_in, range_out=range_out) for x in xx]


def plot_model_and_samples(func: MultiFidelityFunction, kernel: str,
                           scaling_option: str, instance: Instance) -> None:
    """Create a multi-fidelity model based on given instance and show a plot of
    the surfaces and sampled points.
    Can be used for 1D or 2D functions."""

    if func.ndim not in [1, 2]:
        raise ValueError(f"Dimensionality case.func.ndim={func.ndim} not supported by"
                         f"plot_model_and_samples. Only 1D and 2D are supported")

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


def create_model_error_grid(
        func: MultiFidelityFunction,
        instances: Sequence[Instance],
        mfbo_options: Dict[str, Any],
        save_dir: Path,
        extra_attributes=dict(),
        plot_1d: bool=False,
        record_values: bool=False
) -> None:
    """Create a grid of model errors for the given MFF-function case at the
    given list of instances.
    The results are saved in a NETCDF .nc file at the specified `save_dir`"""

    start = datetime.now()
    print(f"Timestamp: {start}")
    print(f"Starting case {func}")
    print(f"{len(instances)} instances passed in")

    Results = namedtuple('Results', 'mses r2 values')

    # Determine unique output path for this experiment
    surr_name = repr_surrogate_name(mfbo_options)
    fname = standardize_fname_for_file(func.name)
    output_path = save_dir / f"{surr_name}-{func.ndim}d-{fname}.nc"

    # Don't redo any prior data that already exists
    if output_path.exists():
        print(f"existing file '{output_path.name}' found, loading instances...")
        num_orig_instances = len(instances)
        with xr.open_mfdataset(f'{output_path}*') as ds:
            with ds['mses'].load() as da:
                instances = filter_instances(instances, da.sel(model='high_hier'))
                print(f"{len(instances)} out of {num_orig_instances} instances left to do")

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
        if record_values:
            values = [model.predict(mfbo.test_sample).flatten()
                      for model in [mfbo.models['high'],
                                    mfbo.direct_models['high'],
                                    mfbo.models['low']]
                      ]
        else:
            values = None

        if plot_1d:
            X = np.linspace(0, 1, 1001).reshape((-1, 1))
            plt.scatter(low_x, low_y, s=20)
            plt.scatter(high_x, high_y, s=15)
            plt.plot(X, mfbo.models['high'].predict(X))
            plt.plot(X, func.high(X))
            plt.savefig(save_dir / f'1d-forrester-visualization-{num_high}-{num_low}.png')
            plt.close()

        # Store the results
        results.append(Results(mses, r2s, values))

    print(f'{len(instances)}/{len(instances)}')



    # Create attributes dictionary
    attributes = dict(experiment='create_model_error_grid',
                      function=func.name,
                      ndim=func.ndim,
                      kernel=mfbo_options.get('kernel', 'N/A'),
                      surrogate_name=mfbo_options.get('surrogate_name', 'Kriging'),
                      scaling=mfbo_options['scaling'],
                      **extra_attributes,
                      )

    ## Iteration finished, arranging data into xr.Dataset
    output = results_to_dataset(results, instances, mfbo_options, attributes,
                                record_values=record_values)

    store_output(output, output_path)

    end = datetime.now()
    print(f"Ended case {func} at {end}\n"
          f"Time spent: {str(end - start)}")


def create_resampling_error_grid(
        func: MultiFidelityFunction,
        DoE_spec: Tuple[int, int],
        instances: Sequence[Instance],
        mfbo_options: Dict[str, Any],
        save_dir: Path,
        extra_attributes=dict(),
        record_values: bool=False
) -> None:
    """Create a grid of model errors for the given MFF-function at the
    given list of instances, with all data for training the model being based
    on an initial given DoE specification.
    The results are saved in a NETCDF .nc file at the specified `save_dir`"""

    start = datetime.now()
    print(f"Starting case {func} at {start}")

    Results = namedtuple('Results', 'mses r2 values')

    # Determine unique output path for this experiment
    surr_name = repr_surrogate_name(mfbo_options)
    doe_high, doe_low = DoE_spec

    fname = standardize_fname_for_file(func.name)
    output_path = save_dir / f"{surr_name}-{func.ndim}d-{fname}-sub{doe_high}-{doe_low}.nc"

    # Don't redo any prior data that already exists
    if output_path.exists():
        print(f"existing file '{output_path.name}' found, loading instances...")
        num_orig_instances = len(instances)
        with xr.open_mfdataset(f'{output_path}*') as ds:
            with ds['mses'].load() as da:
                instances = filter_instances(instances, da.sel(model='high_hier'))
                print(f"{len(instances)} out of {num_orig_instances} instances left to do")

    # Return early if there is nothing left to do
    if not instances:
        print('Nothing to do...')
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
        (high_x, low_x), _ = split_bi_fidelity_doe(DoE, num_high, num_low)
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
        if record_values:
            values = [model.predict(mfbo.test_sample).flatten()
                      for model in [mfbo.models['high'],
                                    mfbo.direct_models['high'],
                                    mfbo.models['low']]
                      ]
        else:
            values = None

        # Store the results
        results.append(Results(mses, r2s, values))

    print(f'{len(instances)}/{len(instances)}')



    # Create attributes dictionary
    attributes = dict(experiment='create_resampling_error_grid',
                      doe=f"{doe_high},{doe_low}",
                      function=func.name,
                      ndim=func.ndim,
                      kernel=mfbo_options.get('kernel', 'N/A'),
                      surrogate_name=mfbo_options.get('surrogate_name', 'Kriging'),
                      scaling=mfbo_options['scaling'],
                      **extra_attributes,
                      )

    ## Iteration finished, arranging data into xr.Dataset
    output = results_to_dataset(results, instances, mfbo_options, attributes,
                                record_values=record_values)


    store_output(output, output_path)

    end = datetime.now()
    print(f"Ended case {func} at {end}\n"
          f"Time spent: {str(end - start)}")


def create_resampling_leftover_error_grid(
        func: MultiFidelityFunction,
        DoE_spec: Tuple[int, int],
        instances: Sequence[Instance],
        mfbo_options: Dict[str, Any],
        save_dir: Path,
        seed_offset=0,
        extra_attributes=dict(),
        record_values: bool=False
) -> None:
    """Create a grid of model errors for the given MFF-function at the
    given list of instances, with all data for training the model being based
    on an initial given DoE specification.
    The results are saved in a NETCDF .nc file at the specified `save_dir`"""

    start = datetime.now()
    print(f"Starting case {func} at {start}")

    Results = namedtuple('Results', 'mses r2 values cv_mses cv_r2')

    # Determine unique output path for this experiment
    surr_name = repr_surrogate_name(mfbo_options)
    doe_high, doe_low = DoE_spec

    fname = standardize_fname_for_file(func.name)
    output_path = save_dir / f"{surr_name}-{func.ndim}d-{fname}-sub{doe_high}-{doe_low}-seed{seed_offset}.nc"

    # Don't redo any prior data that already exists
    if output_path.exists():
        print(f"existing file '{output_path.name}' found, loading instances...")
        num_orig_instances = len(instances)
        with xr.open_mfdataset(f"{output_path}*") as ds:
            da = ds['mses'].load()
            instances = filter_instances(instances, da.sel(model='high_hier'))
            print(f"{len(instances)} out of {num_orig_instances} instances left to do")

    # Return early if there is nothing left to do
    if not instances:
        print('Nothing to do...')
        return

    # Setup some (final) options for the hierarchical model
    mfbo_options['test_sample'] = get_test_sample(func.ndim, save_dir)


    # Create initial DoE
    np.random.seed(20160501 + seed_offset)  # Setting seed for reproducibility
    DoE = bi_fidelity_doe(func.ndim, doe_high, doe_low)
    DoE = scale_to_function(func, DoE)


    results = []
    print('starting loops')
    for i, (num_high, num_low, rep) in enumerate(instances):
        if i % 100 == 0:
            print(f'{i}/{len(instances)}')

        set_seed_by_instance(num_high, num_low, rep)

        # Create sub-sampled Multi-Fidelity DoE in- and output according to instance specification
        selected, test = split_bi_fidelity_doe(DoE, num_high, num_low)
        high_y, low_y = func.high(selected.high), \
                        func.low(selected.low)

        # Create an archive from the MF-function and MF-DoE data
        archive = mlcs.CandidateArchive.from_multi_fidelity_function(func, ndim=func.ndim)
        archive.addcandidates(selected.low, low_y, fidelity='low')
        archive.addcandidates(selected.high, high_y, fidelity='high')

        # (Automatically) Create the hierarchical model
        mfbo = mlcs.MultiFidelityBO(func, archive, **mfbo_options)

        test_high_y, test_low_y = func.high(test.high), \
                                  func.low(test.low)

        models = [
            mfbo.models['high'],
            mfbo.direct_models['high'],
            mfbo.models['low']
        ]
        test_samples = [
            (test.high, test_high_y),
            (test.high, test_high_y),
            (test.low, test_low_y),
        ]

        mses = mfbo.getMSE()
        r2s = mfbo.getR2()
        if record_values:
            values = [
                model.predict(mfbo.test_sample).flatten()
                for model in models
            ]
        else:
            values = None

        # Get the results we're interested in from the model for this instance
        cv_mses = mfbo.MSECollection(*[
            mean_squared_error(y_true, model.predict(x_pred))
            for (x_pred, y_true), model in zip(test_samples, models)
        ])
        cv_r2s = mfbo.R2Collection(*[
            r2_score(y_true, model.predict(x_pred))
            for (x_pred, y_true), model in zip(test_samples, models)
        ])

        # Store the results
        results.append(Results(mses, r2s, values, cv_mses, cv_r2s))

    print(f'{len(instances)}/{len(instances)}')



    # Create attributes dictionary
    attributes = dict(experiment='create_resampling_leftover_error_grid',
                      doe=f"{doe_high},{doe_low}",
                      function=func.name,
                      ndim=func.ndim,
                      kernel=mfbo_options.get('kernel', 'N/A'),
                      surrogate_name=mfbo_options.get('surrogate_name', 'Kriging'),
                      scaling=mfbo_options['scaling'],
                      seed_offset=seed_offset,
                      **extra_attributes,
                      )

    ## Iteration finished, arranging data into an xr.Dataset
    output = results_to_dataset(results, instances, mfbo_options, attributes,
                                record_values=record_values)

    store_output(output, output_path)

    end = datetime.now()
    print(f"Ended case {func} at {end}\n"
          f"Time spent: {str(end - start)}")


def store_output(output: xr.Dataset, output_path: Path):
    """Store output in netcdf .nc file. If previous file(s) exist, store
    in temporary file first, then combine and store combined if memory allows.
    If not, leaves temporary files and prints a warning.

    Note that attributes will be overwritten/updated
    """
    if output_path.exists():
        output.to_netcdf(get_tmp_path(output_path))
        try:
            with xr.open_mfdataset(f'{output_path}*') as output:
                output = output.load()
        except MemoryError:
            warn('MemoryError encountered while merging new output with previous files. '
                 'Temporary files will remain and can be merged later')
            return

    output.to_netcdf(output_path)

    ## remove temporary files
    for file in output_path.parent.glob(f'{output_path.name}TMP*'):
        file.unlink()


def results_to_dataset(results: List[NamedTuple], instances: Sequence[Instance],
                       mfbo_options: Dict[str, Any], attributes: Dict[str, Any],
                       record_values=False) -> xr.Dataset:
    """Manually creating numpy arrays to store the data for eventual
    reading in as XArray DataArray/DataSet"""

    # Hardcoded model names
    models = ['high_hier', 'high', 'low']
    # Get lists of unique entries per instance parameter
    n_highs, n_lows, reps = map(uniquify, zip(*instances))
    n_test_samples = mfbo_options['test_sample'].shape[0]

    common_shape = (len(n_highs), len(n_lows), len(reps), len(models))
    arrays = {}

    # Create empty numpy arrays with the correct minimal size according
    for name, example in zip(results[0]._fields, results[0]):
        if np.array(example).size > len(models):
            shape = (*common_shape, n_test_samples)
        else:
            shape = common_shape
        arrays[name] = np.full(shape, np.nan)

    # Create explicit indices at which values have to be inserted in numpy arrays
    indices = indexify_instances(instances)

    for index, result in zip(indices, results):
        for name, values in zip(result._fields, result):
            arrays[name][index] = values

    all_dims = ['n_high', 'n_low', 'rep', 'model', 'idx']
    dataset = xr.Dataset(
        data_vars={
            name: (all_dims[:values.ndim], values, attributes)
            for name, values in arrays.items()},
        coords={
            'n_high': n_highs,
            'n_low': n_lows,
            'rep': reps,
            'model': models,
            'idx': range(n_test_samples),
        }
    )

    if not record_values:
        dataset = dataset.drop_vars(['values', 'idx'])

    return dataset


def standardize_fname_for_file(name: str) -> str:
    """Enforce standardized formatting for filenames"""
    if 'adjustable' in name.lower():
        fname, param = parse('{} {:f}', name)
        name = f'{fname} {param:.2f}'
    return name.replace(' ', '-')


def create_subsampling_error_grid(
        archive: mlcs.CandidateArchive,
        num_reps: int=50,
        interval: int=2,
        func=None
) -> xr.DataArray:
    """Create an error grid through subsampling from the given archive

    :param archive:  `mlcs.CandidateArchive` containing all available evaluated candidates
    :param num_reps: Number of independent repetitions for each size
    :param interval: Interval at which to fill the error grid
    :param func:     Multi-fidelity function to re-evaluate known candidates  #TODO: remove argument
    :return:         `xr.DataArray` ErrorGrid
    """

    if not func:
        # TODO: implement that the original function is not re-evaluated
        # but that values from `archive` are re-used.
        raise NotImplementedError

    highs = archive.getcandidates(fidelity='high').candidates
    lows = archive.getcandidates(fidelity='low').candidates
    DoE = BiFidelityDoE(highs, lows)

    max_num_high = len(highs)
    max_num_low = len(lows)

    error_grid = np.full((max_num_high+1, max_num_low+1, num_reps+1, 3), np.nan)

    instances = [
        (h, l, r)
        for h, l, r in product(range(2, max_num_high, interval),
                               range(3, max_num_low, interval),
                               range(num_reps))
        if h < l
    ]

    for i, (num_high, num_low, rep) in enumerate(instances):

        if i % 1_000 == 0:
            print(f"{i}/{len(instances)}")

        # Create sub-sampled Multi-Fidelity DoE in- and output according to instance specification
        (high_x, low_x), _ = split_bi_fidelity_doe(DoE, num_high, num_low)
        high_y, low_y = func.high(high_x), \
                        func.low(low_x)

        # Create an archive from the MF-function and MF-DoE data
        arch = mlcs.CandidateArchive.from_multi_fidelity_function(func, ndim=func.ndim)
        arch.addcandidates(low_x, low_y, fidelity='low')
        arch.addcandidates(high_x, high_y, fidelity='high')

        # (Automatically) Create the hierarchical model
        mfbo = mlcs.MultiFidelityBO(func, arch)

        # Get the results we're interested in from the model for this instance
        error_grid[num_high, num_low, rep] = mfbo.getMSE()

    models = ['high_hier', 'high', 'low']
    return xr.DataArray(
        error_grid,
        dims=['n_high', 'n_low', 'rep', 'model'],
        coords={
            'n_high': range(max_num_high+1),
            'n_low': range(max_num_low+1),
            'rep': range(num_reps+1),
            'model': models,
        },
    )


def get_tmp_path(path: Path) -> Path:
    """Return a non-existant path based on the given one"""
    if not path.exists():
        return path

    name = path.name

    tmp_path, tmp_count = path, 0
    while tmp_path.exists():
        tmp_path = path.with_name(f"{name}TMP{tmp_count}")
        tmp_count += 1

    return tmp_path
