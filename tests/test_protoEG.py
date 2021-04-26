import mf2
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xarray as xr

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


def test_protoEG():

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
