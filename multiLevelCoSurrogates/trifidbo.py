import numpy as np
import xarray as xr

from pprint import pprint

from pyDOE import lhs
from multiLevelCoSurrogates.config import fit_funcs
from multiLevelCoSurrogates.local import base_dir
from multiLevelCoSurrogates.Utils import select_subsample
from multiLevelCoSurrogates.CandidateArchive import CandidateArchive
from multiLevelCoSurrogates.Surrogates import Kriging, HierarchicalSurrogate


def run(multi_fid_func):
    ndim = multi_fid_func.ndim
    fidelities = list(multi_fid_func.fidelity_names)
    print(fidelities)

    num_samples = dict(zip(fidelities, [3, 5, 10]))
    print(num_samples)

    samples = {'low': lhs(n=ndim, samples=num_samples['low'])}
    samples['medium'] = select_subsample(samples['low'].T, num_samples['medium']).T
    samples['high'] = select_subsample(samples['medium'].T, num_samples['high']).T

    archive_fidelities = fidelities + [f'{a}-{b}' for a, b in zip(fidelities, fidelities[1:])]
    print(archive_fidelities)
    archive = CandidateArchive(ndim=ndim, fidelities=archive_fidelities)
    for fidelity in fidelities:
        print(fidelity)
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




if __name__ == '__main__':
    run(fit_funcs['himmelblau'])
