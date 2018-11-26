import numpy as np
import xarray as xr
import bayes_opt as bo

from pprint import pprint, PrettyPrinter
from functools import partial

from multiLevelCoSurrogates.bo import gpplot, createsurfaces, plotsurfaces, ScatterPoints

from pyDOE import lhs
from multifidelityfunctions import TriFidelityFunction
from multiLevelCoSurrogates.config import fit_funcs
from multiLevelCoSurrogates.local import base_dir
from multiLevelCoSurrogates.Utils import select_subsample, linearscaletransform, ValueRange
from multiLevelCoSurrogates.CandidateArchive import CandidateArchive
from multiLevelCoSurrogates.Surrogates import Kriging, HierarchicalSurrogate

import tracemalloc

tracemalloc.start()



Pprint = PrettyPrinter(width=200)

def run(multi_fid_func, num_iters=13):

    ndim = multi_fid_func.ndim
    fidelities = list(multi_fid_func.fidelity_names)
    bounds = np.array([multi_fid_func.l_bound, multi_fid_func.u_bound], dtype=np.float)

    schema = [4,2,1]

    samples = create_sample_set(ndim, zip(fidelities, [3,5,10]),
                                desired_range=ValueRange(*bounds))

    archive_fidelities = fidelities + [f'{a}-{b}' for a, b in zip(fidelities, fidelities[1:])]
    archive = CandidateArchive(ndim=ndim, fidelities=archive_fidelities)
    for fidelity in fidelities:
        archive.addcandidates(
            samples[fidelity],
            getattr(multi_fid_func, fidelity)(samples[fidelity]),
            fidelity=fidelity
        )

    pprint(archive.data)

    low_model = Kriging(archive, num_points=None, fidelity='low')
    mid_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=low_model,
                                      candidate_archive=archive, fidelities=fidelities[1:3])
    high_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=mid_model,
                                       candidate_archive=archive, fidelities=fidelities[0:2])

    high_model.retrain()

    utility = bo.helpers.UtilityFunction(kind='ei', kappa=2.576, xi=1.0).utility
    acq_max = partial(bo.helpers.acq_max,
                      ac=utility, gp=high_model, random_state=np.random.RandomState(),
                      bounds=bounds.T, n_warmup=1000, n_iter=50)

    functions_to_plot = [
        multi_fid_func.high, multi_fid_func.medium, multi_fid_func.low,

        partial(gpplot, func=high_model.predict),
        partial(gpplot, func=mid_model.predict),
        partial(gpplot, func=low_model.predict),

        partial(gpplot, func=high_model.predict, return_std=True),
        partial(gpplot, func=mid_model.predict, return_std=True),
        partial(gpplot, func=low_model.predict, return_std=True),

        lambda x: utility(x, gp=high_model, y_max=archive.max['high']),
        lambda x: utility(x, gp=mid_model, y_max=archive.max['medium']),
        lambda x: utility(x, gp=low_model, y_max=archive.max['low']),
    ]
    titles = [
        'high', 'medium', 'low',
        'high model', 'medium model', 'low model',
        'high std', 'medium std', 'low std',
        'acq_high', 'acq_medium', 'acq_low',
    ]

    for iteration in range(num_iters):
        next_point = acq_max(y_max=archive.max['high'])

        for interval, fidelity in zip(schema, fidelities):
            if iteration % interval == 0:
                next_value = getattr(multi_fid_func, fidelity)(next_point)
                archive.addcandidate(next_point, next_value, fidelity=fidelity)
        print(f'iteration: {iteration} | archive_size: {len(archive)} | next point: {next_point}')
        high_model.retrain()

        red_dot = {'marker': '.', 'color': 'red'}
        blue_circle = {'marker': 'o', 'facecolors': 'none', 'color': 'blue'}
        green_cross = {'marker': '+', 'color': 'green'}
        surfaces = createsurfaces(functions_to_plot, l_bound=multi_fid_func.l_bound, u_bound=multi_fid_func.u_bound)
        points = [
            [ScatterPoints(*archive.getcandidates(fidelity='high'), style=red_dot)],
            [ScatterPoints(*archive.getcandidates(fidelity='medium'), style=blue_circle)],
            [ScatterPoints(*archive.getcandidates(fidelity='low'), style=green_cross)],
        ]*4

        # if (iteration%4) == 0:
        plotsurfaces(surfaces, all_points=points, titles=titles, as_3d=False, shape=(4,3))

        print('MEMORY USAGE:')
        snapshot = tracemalloc.take_snapshot()
        for stat in snapshot.statistics('lineno')[:5]:
            print(stat)
        print()



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

    old_hm = fit_funcs['himmelblau']
    hm = TriFidelityFunction(
        u_bound=np.array(old_hm.u_bound)*bound_factor, l_bound=np.array(old_hm.l_bound)*bound_factor,
        high=lambda x: -old_hm.high(x), medium=lambda x: -old_hm.medium(x), low=lambda x: -old_hm.low(x)
    )

    run(hm)
