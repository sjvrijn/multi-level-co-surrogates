import mf2
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xarray as xr

from pyprojroot import here
import sys
module_path = str(here('scripts/experiments'))
if module_path not in sys.path:
    sys.path.append(module_path)

import experiments as exp
import multiLevelCoSurrogates as mlcs



def prepare_DoE(func, nh=3, nl=4):
    np.random.seed(20160501)  # Setting seed for reproducibility
    init_DoE = exp.bi_fidelity_doe(func.ndim, nh, nl)
    DoE = exp.scale_to_function(func, init_DoE)
    return DoE


def get_experiment_subsampled_EG(func, DoE, instances):

    results = []
    for i, (num_high, num_low, rep) in enumerate(instances):

        mlcs.set_seed_by_instance(num_high, num_low, rep)

        # Create sub-sampled Multi-Fidelity DoE in- and output according to instance specification
        train, test = exp.split_bi_fidelity_doe(DoE, num_high, num_low)
        train_high_y, train_low_y = func.high(train.high), \
                                    func.low(train.low)

        # Create an archive from the MF-function and MF-DoE data
        archive = mlcs.CandidateArchive.from_multi_fidelity_function(func, ndim=func.ndim)
        archive.addcandidates(train.low, train_low_y, fidelity='low')
        archive.addcandidates(train.high, train_high_y, fidelity='high')

        # (Automatically) Create the hierarchical model
        mfbo = mlcs.MultiFidelityBO(func, archive, scaling='off', kernel='Matern')

        mses = mfbo.getMSE()

        test_high_y = func.high(test.high)
        cv_mses = mean_squared_error(test_high_y, mfbo.models['high'].predict(test.high))

        # Store the results
        results.append((num_high, num_low, rep, 'high_hier', cv_mses, mses[0]))

    columns = ['n_high', 'n_low', 'rep', 'model', 'mses', 'orig_mses']

    tmp_df = pd.DataFrame.from_records(results, columns=columns, index=columns[:4])
    return xr.Dataset.from_dataframe(tmp_df)


def get_subsampled_protoEG(archive, num_reps):

    eg = mlcs.ProtoEG(archive, num_reps=num_reps)
    eg.subsample_errorgrid()
    return eg


def test_experiment():

    func = mf2.currin
    num_reps = 1
    DoE = prepare_DoE(func)

    spec = mlcs.InstanceSpec(len(DoE[0])-1, len(DoE[1]), num_reps=num_reps)
    instances = list(spec.instances)
    eg2 = get_experiment_subsampled_EG(func, DoE, instances=instances)

    np.testing.assert_allclose(
        [0.003953943893931256],
        eg2['mses'].sel(model='high_hier').values.flatten()[0],
    )


def test_protoEG_subsample_errorgrid_create():

    func = mf2.currin
    num_reps = 1
    DoE_high, DoE_low = prepare_DoE(func)

    archive = mlcs.CandidateArchive.from_multi_fidelity_function(func)
    archive.addcandidates(DoE_high, func.high(DoE_high), fidelity='high')
    archive.addcandidates(DoE_low, func.low(DoE_low), fidelity='low')

    proto_eg = get_subsampled_protoEG(archive, num_reps)
    eg1 = proto_eg.error_grid

    np.testing.assert_allclose(
        [0.003953943893931256],
        eg1['mses'].sel(model='high_hier').values.flatten()[0],
    )



def test_protoEG_subsample_errorgrid_update_low():

    func = mf2.currin
    num_reps = 1
    DoE_high, DoE_low = prepare_DoE(func)
    archive = mlcs.CandidateArchive.from_multi_fidelity_function(func)
    archive.addcandidates(DoE_high, func.high(DoE_high), fidelity='high')
    archive.addcandidates(DoE_low, func.low(DoE_low), fidelity='low')
    proto_eg = get_subsampled_protoEG(archive, num_reps)

    np.random.seed(0)
    new_sample = np.random.rand(1,func.ndim)
    proto_eg.update_errorgrid_with_sample(new_sample, func.low(new_sample), fidelity='low')
    print(proto_eg.error_grid)
    assert len(archive) == len(DoE_low) + 1


def test_protoEG_subsample_errorgrid_update_high():

    func = mf2.currin
    num_reps = 1
    DoE_high, DoE_low = prepare_DoE(func)

    archive = mlcs.CandidateArchive.from_multi_fidelity_function(func)
    archive.addcandidates(DoE_high, func.high(DoE_high), fidelity='high')
    archive.addcandidates(DoE_low, func.low(DoE_low), fidelity='low')
    old_len = len(archive)

    proto_eg = get_subsampled_protoEG(archive, num_reps)

    np.random.seed(0)
    non_high = set(tuple(c) for c in DoE_low) - set(tuple(c) for c in DoE_high)
    print(non_high)
    new_sample = np.array(next(iter(non_high))).reshape(1, -1) # just take 1 element
    print(new_sample)
    proto_eg.update_errorgrid_with_sample(new_sample, func.high(new_sample), fidelity='high')
    print(proto_eg.error_grid)
    assert archive.count('high') == len(DoE_high) + 1
