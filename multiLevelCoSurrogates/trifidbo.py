import numpy as np
import pandas as pd
import bayes_opt as bo

from functools import partial
from collections import namedtuple
from sklearn.metrics import mean_squared_error
from pyDOE import lhs


from multifidelityfunctions import TriFidelityFunction
from multiLevelCoSurrogates.bo import gpplot, ScatterPoints
from multiLevelCoSurrogates.config import fit_funcs
from multiLevelCoSurrogates.local import base_dir
from multiLevelCoSurrogates.CandidateArchive import CandidateArchive
from multiLevelCoSurrogates.Utils import select_subsample, linearscaletransform, \
    sample_by_function, createsurfaces, plotsurfaces, ValueRange
from multiLevelCoSurrogates.Surrogates import Kriging, HierarchicalSurrogate


import tracemalloc
tracemalloc.start()


MSECollection = namedtuple('MSECollection', ['high_hier', 'high', 'medium_hier', 'medium', 'low'])
MSERecord = namedtuple('MSERecord', ['repetition', 'iteration',
                                     *('mse_' + mse for mse in MSECollection._fields)])

def flatten(iterable):
    return [x for y in iterable for x in y]


def run(multi_fid_func, *, num_iters=100, repetition_idx=0, do_plot=False):

    ndim = multi_fid_func.ndim
    fidelities = list(multi_fid_func.fidelity_names)
    bounds = np.array([multi_fid_func.l_bound, multi_fid_func.u_bound], dtype=np.float)
    input_range = ValueRange(*bounds)
    output_range = ValueRange(-450, 0)

    schema = [4,2,1]

    samples = create_sample_set(ndim, zip(fidelities, [5,8,13]),
                                desired_range=input_range)

    n_samples = 1000
    test_sample = sample_by_function(multi_fid_func.high, n_samples=n_samples, ndim=ndim,
                                     range_in=input_range, range_out=output_range)

    mse_tester = {
        fid: partial(mean_squared_error,
                     y_pred=getattr(multi_fid_func, fid)(test_sample))
        for fid in fidelities
    }

    archive_fidelities = fidelities + [f'{a}-{b}' for a, b in zip(fidelities, fidelities[1:])]
    archive = CandidateArchive(ndim=ndim, fidelities=archive_fidelities)
    for fidelity in fidelities:
        archive.addcandidates(
            samples[fidelity],
            getattr(multi_fid_func, fidelity)(samples[fidelity]),
            fidelity=fidelity
        )

    low_model = Kriging(archive, num_points=None, fidelity='low')
    medium_model = Kriging(archive, num_points=None, fidelity='medium')
    high_model = Kriging(archive, num_points=None, fidelity='high')

    medium_hier_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=low_model,
                                              candidate_archive=archive, fidelities=fidelities[1:3])
    high_hier_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=medium_hier_model,
                                            candidate_archive=archive, fidelities=fidelities[0:2])

    high_hier_model.retrain()
    medium_model.retrain()
    high_model.retrain()

    utility = bo.helpers.UtilityFunction(kind='ei', kappa=2.576, xi=1.0).utility
    acq_max = partial(bo.helpers.acq_max,
                      ac=utility, gp=high_hier_model, random_state=np.random.RandomState(),
                      bounds=bounds.T, n_warmup=1000, n_iter=50)

    functions_to_plot = [
        multi_fid_func.high, multi_fid_func.medium, multi_fid_func.low,

        partial(gpplot, func=high_hier_model.predict),
        partial(gpplot, func=medium_hier_model.predict),
        partial(gpplot, func=low_model.predict),

        partial(gpplot, func=high_hier_model.predict, return_std=True),
        partial(gpplot, func=medium_hier_model.predict, return_std=True),
        partial(gpplot, func=low_model.predict, return_std=True),

        lambda x: utility(x, gp=high_hier_model, y_max=archive.max['high']),
        lambda x: utility(x, gp=medium_hier_model, y_max=archive.max['medium']),
        lambda x: utility(x, gp=low_model, y_max=archive.max['low']),
    ]
    titles = [
        'high', 'medium', 'low',
        'high model', 'medium model', 'low model',
        'high std', 'medium std', 'low std',
        'acq_high', 'acq_medium', 'acq_low',
    ]

    records = []

    for iteration in range(num_iters):
        next_point_low = acq_max(y_max=archive.max['high'])
        next_point_medium = next_point_high = None

        next_value_low = multi_fid_func.low(next_point_low)
        archive.addcandidate(next_point_low, next_value_low, fidelity='low')
        high_hier_model.retrain()

        if iteration % schema[1] == 0:
            candidates_low = archive.getcandidates(fidelity='low').candidates
            candidates_medium = archive.getcandidates(fidelity='medium').candidates
            interesting_candidates = list({tuple(x.tolist()) for x in candidates_low}
                                          - {tuple(y.tolist()) for y in candidates_medium})
            predicted_values = high_hier_model.predict(np.array(interesting_candidates))

            next_point_medium = list(interesting_candidates[np.argmax(predicted_values)])
            next_value_medium = multi_fid_func.medium(next_point_medium)
            archive.addcandidate(next_point_medium, next_value_medium, fidelity='medium')
            high_hier_model.retrain()

        if iteration % schema[0] == 0:
            candidates_medium = archive.getcandidates(fidelity='medium').candidates
            candidates_high = archive.getcandidates(fidelity='high').candidates

            interesting_candidates = list({tuple(x.tolist()) for x in candidates_medium}
                                          - {tuple(y.tolist()) for y in candidates_high})
            predicted_values = high_hier_model.predict(np.array(interesting_candidates))

            next_point_high = list(interesting_candidates[np.argmax(predicted_values)])
            next_value_high = multi_fid_func.medium(next_point_high)
            archive.addcandidate(next_point_high, next_value_high, fidelity='high')
            high_hier_model.retrain()

        print(f'iteration: {iteration} | archive_size: {len(archive)} | next point: {next_point_low} {next_point_medium} {next_point_high}')
        medium_model.retrain()
        high_model.retrain()

        mses = MSECollection(
            mse_tester['high'](high_hier_model.predict(test_sample)),
            mse_tester['high'](high_model.predict(test_sample)),
            mse_tester['medium'](medium_hier_model.predict(test_sample)),
            mse_tester['medium'](medium_model.predict(test_sample)),
            mse_tester['low'](low_model.predict(test_sample)),
        )
        records.append(MSERecord(repetition_idx, iteration, *mses))

        if do_plot:
            red_dot = {'marker': '.', 'color': 'red'}
            blue_circle = {'marker': 'o', 'facecolors': 'none', 'color': 'blue'}
            green_cross = {'marker': '+', 'color': 'green'}
            surfaces = createsurfaces(functions_to_plot, l_bound=multi_fid_func.l_bound, u_bound=multi_fid_func.u_bound)
            points = [
                [ScatterPoints(*archive.getcandidates(fidelity='high'), style=red_dot)],
                [ScatterPoints(*archive.getcandidates(fidelity='medium'), style=blue_circle)],
                [ScatterPoints(*archive.getcandidates(fidelity='low'), style=green_cross)],
            ]*4

            plotsurfaces(surfaces, all_points=points, titles=titles, as_3d=False, shape=(4,3))

    return pd.DataFrame(records)


def run_hm(multi_fid_func, *, num_iters=100, repetition_idx=0, do_plot=False):

    ndim = multi_fid_func.ndim
    fidelities = ['high', 'medium']
    bounds = np.array([multi_fid_func.l_bound, multi_fid_func.u_bound], dtype=np.float)
    input_range = ValueRange(*bounds)
    output_range = ValueRange(-450, 0)

    schema = [2,1]

    samples = create_sample_set(ndim, zip(fidelities, [5,8]),
                                desired_range=input_range)

    n_samples = 1000
    test_sample = sample_by_function(multi_fid_func.high, n_samples=n_samples, ndim=ndim,
                                     range_in=input_range, range_out=output_range)

    mse_tester = {
        fid: partial(mean_squared_error,
                     y_pred=getattr(multi_fid_func, fid)(test_sample))
        for fid in fidelities
    }

    archive_fidelities = fidelities + [f'{a}-{b}' for a, b in zip(fidelities, fidelities[1:])]
    archive = CandidateArchive(ndim=ndim, fidelities=archive_fidelities)
    for fidelity in fidelities:
        archive.addcandidates(
            samples[fidelity],
            getattr(multi_fid_func, fidelity)(samples[fidelity]),
            fidelity=fidelity
        )

    medium_model = Kriging(archive, num_points=None, fidelity='medium')
    high_model = Kriging(archive, num_points=None, fidelity='high')

    high_hier_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=medium_model,
                                            candidate_archive=archive, fidelities=fidelities[0:2])

    high_hier_model.retrain()
    high_model.retrain()

    utility = bo.helpers.UtilityFunction(kind='ei', kappa=2.576, xi=1.0).utility
    acq_max = partial(bo.helpers.acq_max,
                      ac=utility, gp=high_hier_model, random_state=np.random.RandomState(),
                      bounds=bounds.T, n_warmup=1000, n_iter=50)

    functions_to_plot = [
        multi_fid_func.high, multi_fid_func.medium,

        partial(gpplot, func=high_hier_model.predict),
        partial(gpplot, func=medium_model.predict),

        partial(gpplot, func=high_hier_model.predict, return_std=True),
        partial(gpplot, func=medium_model.predict, return_std=True),

        lambda x: utility(x, gp=high_hier_model, y_max=archive.max['high']),
        lambda x: utility(x, gp=medium_model, y_max=archive.max['medium']),
    ]
    titles = [
        'high', 'medium',
        'high model', 'medium model',
        'high std', 'medium std',
        'acq_high', 'acq_medium',
    ]

    records = []

    for iteration in range(num_iters):
        next_point_low = acq_max(y_max=archive.max['high'])
        next_point_medium = next_point_high = None

        next_value_medium = multi_fid_func.medium(next_point_low)
        archive.addcandidate(next_point_low, next_value_medium, fidelity='medium')
        high_hier_model.retrain()

        if iteration % schema[0] == 0:
            candidates_medium = archive.getcandidates(fidelity='medium').candidates
            candidates_high = archive.getcandidates(fidelity='high').candidates

            interesting_candidates = list({tuple(x.tolist()) for x in candidates_medium}
                                          - {tuple(y.tolist()) for y in candidates_high})
            predicted_values = high_hier_model.predict(np.array(interesting_candidates))

            next_point_high = list(interesting_candidates[np.argmax(predicted_values)])
            next_value_high = multi_fid_func.medium(next_point_high)
            archive.addcandidate(next_point_high, next_value_high, fidelity='high')
            high_hier_model.retrain()

        print(f'iteration: {iteration} | archive_size: {len(archive)} | next point: {next_point_low} {next_point_medium} {next_point_high}')
        medium_model.retrain()
        high_model.retrain()

        mses = MSECollection(
            mse_tester['high'](high_hier_model.predict(test_sample)),
            mse_tester['high'](high_model.predict(test_sample)),
            0,
            mse_tester['medium'](medium_model.predict(test_sample)),
            0,
        )
        records.append(MSERecord(repetition_idx, iteration, *mses))

        if do_plot:
            red_dot = {'marker': '.', 'color': 'red'}
            blue_circle = {'marker': 'o', 'facecolors': 'none', 'color': 'blue'}
            surfaces = createsurfaces(functions_to_plot, l_bound=multi_fid_func.l_bound, u_bound=multi_fid_func.u_bound)
            points = [
                [ScatterPoints(*archive.getcandidates(fidelity='high'), style=red_dot)],
                [ScatterPoints(*archive.getcandidates(fidelity='medium'), style=blue_circle)],
            ]*4

            plotsurfaces(surfaces, all_points=points, titles=titles, as_3d=False, shape=(4,2))

    return pd.DataFrame(records)


def run_hl(multi_fid_func, *, num_iters=100, repetition_idx=0, do_plot=False):

    ndim = multi_fid_func.ndim
    fidelities = ['high', 'low']
    bounds = np.array([multi_fid_func.l_bound, multi_fid_func.u_bound], dtype=np.float)
    input_range = ValueRange(*bounds)
    output_range = ValueRange(-450, 0)

    schema = [4,1]

    samples = create_sample_set(ndim, zip(fidelities, [5,13]),
                                desired_range=input_range)

    n_samples = 1000
    test_sample = sample_by_function(multi_fid_func.high, n_samples=n_samples, ndim=ndim,
                                     range_in=input_range, range_out=output_range)

    mse_tester = {
        fid: partial(mean_squared_error,
                     y_pred=getattr(multi_fid_func, fid)(test_sample))
        for fid in fidelities
    }

    archive_fidelities = fidelities + [f'{a}-{b}' for a, b in zip(fidelities, fidelities[1:])]
    archive = CandidateArchive(ndim=ndim, fidelities=archive_fidelities)
    for fidelity in fidelities:
        archive.addcandidates(
            samples[fidelity],
            getattr(multi_fid_func, fidelity)(samples[fidelity]),
            fidelity=fidelity
        )

    low_model = Kriging(archive, num_points=None, fidelity='low')
    high_model = Kriging(archive, num_points=None, fidelity='high')

    high_hier_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=low_model,
                                            candidate_archive=archive, fidelities=fidelities[0:2])

    high_hier_model.retrain()
    high_model.retrain()

    utility = bo.helpers.UtilityFunction(kind='ei', kappa=2.576, xi=1.0).utility
    acq_max = partial(bo.helpers.acq_max,
                      ac=utility, gp=high_hier_model, random_state=np.random.RandomState(),
                      bounds=bounds.T, n_warmup=1000, n_iter=50)

    functions_to_plot = [
        multi_fid_func.high, multi_fid_func.low,

        partial(gpplot, func=high_hier_model.predict),
        partial(gpplot, func=low_model.predict),

        partial(gpplot, func=high_hier_model.predict, return_std=True),
        partial(gpplot, func=low_model.predict, return_std=True),

        lambda x: utility(x, gp=high_hier_model, y_max=archive.max['high']),
        lambda x: utility(x, gp=low_model, y_max=archive.max['low']),
    ]
    titles = [
        'high', 'low',
        'high model', 'low model',
        'high std', 'low std',
        'acq_high', 'acq_low',
    ]

    records = []

    for iteration in range(num_iters):
        next_point_low = acq_max(y_max=archive.max['high'])
        next_point_medium = next_point_high = None

        next_value_low = multi_fid_func.low(next_point_low)
        archive.addcandidate(next_point_low, next_value_low, fidelity='low')
        high_hier_model.retrain()

        if iteration % schema[0] == 0:
            candidates_medium = archive.getcandidates(fidelity='low').candidates
            candidates_high = archive.getcandidates(fidelity='high').candidates

            interesting_candidates = list({tuple(x.tolist()) for x in candidates_medium}
                                          - {tuple(y.tolist()) for y in candidates_high})
            predicted_values = high_hier_model.predict(np.array(interesting_candidates))

            next_point_high = list(interesting_candidates[np.argmax(predicted_values)])
            next_value_high = multi_fid_func.medium(next_point_high)
            archive.addcandidate(next_point_high, next_value_high, fidelity='high')
            high_hier_model.retrain()

        print(f'iteration: {iteration} | archive_size: {len(archive)} | next point: {next_point_low} {next_point_medium} {next_point_high}')
        high_model.retrain()

        mses = MSECollection(
            mse_tester['high'](high_hier_model.predict(test_sample)),
            mse_tester['high'](high_model.predict(test_sample)),
            0,
            0,
            mse_tester['low'](low_model.predict(test_sample)),
        )
        records.append(MSERecord(repetition_idx, iteration, *mses))

        if do_plot:
            red_dot = {'marker': '.', 'color': 'red'}
            green_cross = {'marker': '+', 'color': 'green'}
            surfaces = createsurfaces(functions_to_plot, l_bound=multi_fid_func.l_bound, u_bound=multi_fid_func.u_bound)
            points = [
                [ScatterPoints(*archive.getcandidates(fidelity='high'), style=red_dot)],
                [ScatterPoints(*archive.getcandidates(fidelity='low'), style=green_cross)],
            ]*4

            plotsurfaces(surfaces, all_points=points, titles=titles, as_3d=False, shape=(4,2))

    return pd.DataFrame(records)


def create_sample_set(ndim, size_per_fidelity, desired_range=None):
    size_per_fidelity = iter(sorted(size_per_fidelity, key=lambda x: x[1], reverse=True))
    range_lhs = ValueRange(0, 1)

    fid, size = next(size_per_fidelity)
    sample = lhs(n=ndim, samples=size)
    if desired_range is not None:
        sample = linearscaletransform(sample, range_in=range_lhs, range_out=desired_range)
    samples = {fid: sample}
    for fid, size in size_per_fidelity:
        sample = select_subsample(sample.T, num=size).T
        samples[fid] = sample

    return samples


if __name__ == '__main__':
    np.set_printoptions(linewidth=200)

    bound_factor = 1
    num_repetitions = 15
    num_iterations = [102, 51, 102]
    do_plot = False
    fit_func_names = ['himmelblau_seb', 'himmelblau']


    for name in fit_func_names:

        old_hm = fit_funcs[name]
        hm = TriFidelityFunction(
            u_bound=np.array(old_hm.u_bound) * bound_factor, l_bound=np.array(old_hm.l_bound) * bound_factor,
            high=lambda x: -old_hm.high(x), medium=lambda x: -old_hm.medium(x), low=lambda x: -old_hm.low(x)
        )

        save_name = f'{base_dir}{name}_hml_records.csv'
        for rep in range(num_repetitions):
            print(f'{rep}/{num_repetitions}')
            np.random.seed(rep)
            df = run(hm, num_iters=num_iterations[0], repetition_idx=rep, do_plot=do_plot)

            if rep != 0:
                try:
                    df = pd.concat([pd.read_csv(save_name, index_col='index'), df], ignore_index=True)
                except FileNotFoundError:
                    pass
            df.to_csv(save_name, index_label='index')

        save_name = f'{base_dir}{name}_hm_records.csv'
        for rep in range(num_repetitions):
            print(f'{rep}/{num_repetitions}')
            np.random.seed(rep)
            df = run_hm(hm, num_iters=num_iterations[1], repetition_idx=rep, do_plot=do_plot)

            if rep != 0:
                try:
                    df = pd.concat([pd.read_csv(save_name, index_col='index'), df], ignore_index=True)
                except FileNotFoundError:
                    pass
            df.to_csv(save_name, index_label='index')

        save_name = f'{base_dir}{name}_hl_records.csv'
        for rep in range(num_repetitions):
            print(f'{rep}/{num_repetitions}')
            np.random.seed(rep)
            df = run_hl(hm, num_iters=num_iterations[2], repetition_idx=rep, do_plot=do_plot)

            if rep != 0:
                try:
                    df = pd.concat([pd.read_csv(save_name, index_col='index'), df], ignore_index=True)
                except FileNotFoundError:
                    pass
            df.to_csv(save_name, index_label='index')
