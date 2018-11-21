import numpy as np
import xarray as xr
import bayes_opt as bo

from pprint import pprint, PrettyPrinter
from functools import partial

from multiLevelCoSurrogates.bo import gpplot, createsurfaces, plotsurfaces, ScatterPoints

from pyDOE import lhs
from multiLevelCoSurrogates.config import fit_funcs
from multiLevelCoSurrogates.local import base_dir
from multiLevelCoSurrogates.Utils import select_subsample, linearscaletransform, ValueRange
from multiLevelCoSurrogates.CandidateArchive import CandidateArchive
from multiLevelCoSurrogates.Surrogates import Kriging, HierarchicalSurrogate



Pprint =  PrettyPrinter(width=200)

def run(multi_fid_func, num_iters=100):

    ndim = multi_fid_func.ndim
    fidelities = list(multi_fid_func.fidelity_names)
    bounds = np.array([multi_fid_func.l_bound, multi_fid_func.u_bound], dtype=np.float)

    schema = [9,3,1]

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

    low_model = Kriging(archive, num_points=None, fidelity='low')
    mid_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=low_model,
                                      candidate_archive=archive, fidelities=fidelities[1:3])
    high_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=mid_model,
                                       candidate_archive=archive, fidelities=fidelities[0:2])

    high_model.train()

    util = bo.helpers.UtilityFunction(kind='ei', kappa=2.576, xi=0.5)
    def acq_func(*args, **kwargs):
        return -util.utility(*args, **kwargs)
    acq_max = partial(bo.helpers.acq_max,
                      ac=acq_func, gp=high_model, random_state=np.random.RandomState(),
                      bounds=bounds, n_warmup=1000, n_iter=50)

    functions_to_plot = [
        multi_fid_func.high, multi_fid_func.medium, multi_fid_func.low,

        partial(gpplot, func=high_model.predict),
        partial(gpplot, func=mid_model.predict),
        partial(gpplot, func=low_model.predict),
    ]
    titles = [
        'high', 'medium', 'low',
        'high model', 'medium model', 'low model',
    ]

    for iteration in range(num_iters):
        next_point = acq_max(y_max=archive.max['high'])
        print('next:', next_point)

        for interval, fidelity in zip(schema, fidelities):
            print('\t', iteration, interval, iteration%interval)
            if iteration % interval == 0:
                next_value = getattr(multi_fid_func, fidelity)(next_point)
                archive.addcandidate(next_point, next_value, fidelity=fidelity)
        print(len(archive))
        high_model.train()

        red_dot = {'marker': '.', 'color': 'red'}
        surfaces = createsurfaces(functions_to_plot)
        points = [
            [ScatterPoints(*archive.getcandidates(fidelity='high'), style=red_dot)],
            [ScatterPoints(*archive.getcandidates(fidelity='medium'), style=red_dot)],
            [ScatterPoints(*archive.getcandidates(fidelity='low'), style=red_dot)],
        ]*2

        if (iteration%9) == 0:
            plotsurfaces(surfaces, all_points=points, titles=titles, as_3d=True, shape=(2,3))



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
    run(fit_funcs['himmelblau'])
