import numpy as np
import xarray as xr
import bayes_opt as bo

from pprint import pprint, PrettyPrinter
from functools import partial

from multiLevelCoSurrogates.bo import gpplot, createsurfaces, plotsurfaces

from pyDOE import lhs
from multiLevelCoSurrogates.config import fit_funcs
from multiLevelCoSurrogates.local import base_dir
from multiLevelCoSurrogates.Utils import select_subsample
from multiLevelCoSurrogates.CandidateArchive import CandidateArchive
from multiLevelCoSurrogates.Surrogates import Kriging, HierarchicalSurrogate



Pprint =  PrettyPrinter(width=200)

def run(multi_fid_func, num_iters=100):

    ndim = multi_fid_func.ndim
    fidelities = list(multi_fid_func.fidelity_names)
    bounds = np.array([multi_fid_func.l_bound, multi_fid_func.u_bound], dtype=np.float)

    print(bounds)

    schema = [9,3,1]

    samples = create_sample_set(ndim, zip(fidelities, [3,5,10]))

    archive_fidelities = fidelities + [f'{a}-{b}' for a, b in zip(fidelities, fidelities[1:])]
    archive = CandidateArchive(ndim=ndim, fidelities=archive_fidelities)
    for fidelity in fidelities:
        archive.addcandidates(
            samples[fidelity],
            getattr(multi_fid_func, fidelity)(samples[fidelity]),
            fidelity=fidelity
        )

    # low_model = Kriging(archive, num_points=None, fidelity='low')
    # mid_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=low_model,
    #                                   candidate_archive=archive, fidelities=fidelities[1:3])
    # high_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=mid_model,
    #                                    candidate_archive=archive, fidelities=fidelities[0:2])

    high_model = Kriging(archive, num_points=None, fidelity='low')
    # mid_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=low_model,
    #                                   candidate_archive=archive, fidelities=fidelities[1:3])
    # high_model = HierarchicalSurrogate('Kriging', lower_fidelity_model=mid_model,
    #                                    candidate_archive=archive, fidelities=fidelities[0:2])

    acq_func = bo.helpers.UtilityFunction(kind='ei', kappa=2.576, xi=0.5).utility
    acq_max = partial(bo.helpers.acq_max, ac=acq_func, gp=high_model, random_state=np.random.RandomState(),
                      bounds=bounds, n_warmup=1000, n_iter=50)


    for iteration in range(num_iters):
        high_model.train()
        next_point = acq_max(y_max=archive.max['high'])
        # print(next_point)

        for interval, fidelity in zip(schema, fidelities):
            # print('\t', iteration, interval, iteration%interval)
            if iteration % interval == 0:
                next_value = getattr(multi_fid_func, fidelity)(next_point)
                archive.addcandidate(next_point, next_value, fidelity=fidelity)

        # surfaces = createsurfaces([
        #     partial(gpplot, func=low_model.predict),
        #     partial(gpplot, func=mid_model.predict),
        #     partial(gpplot, func=high_model.predict),
        # ])
        # titles = [
        #     'low model',
        #     'mid model',
        #     'high model',
        # ]
        # plotsurfaces(surfaces, titles=titles)
        print(iteration)
        Pprint.pprint(archive.data)



def create_sample_set(ndim, size_per_fidelity):
    size_per_fidelity = iter(sorted(size_per_fidelity, key=lambda x: x[1], reverse=True))

    fid, size = next(size_per_fidelity)
    sample = lhs(n=ndim, samples=size)
    samples = {fid: sample}
    for fid, size in size_per_fidelity:
        sample = select_subsample(sample.T, num=size).T
        samples[fid] = sample

    return samples


if __name__ == '__main__':
    np.set_printoptions(linewidth=200)
    run(fit_funcs['himmelblau'])
