from hypothesis import given
from hypothesis.strategies import composite, integers
import mf2
import numpy as np
import pandas as pd
from pytest import approx
from scipy.special import binom
from sklearn.metrics import mean_squared_error
import xarray as xr

from pyprojroot import here
import sys
module_path = str(here('scripts/experiments'))
if module_path not in sys.path:
    sys.path.append(module_path)

import multiLevelCoSurrogates as mlcs



def prepare_DoE(func, nh=3, nl=5):
    np.random.seed(20160501)  # Setting seed for reproducibility
    init_DoE = mlcs.bi_fidelity_doe(func.ndim, nh, nl)
    DoE = mlcs.scale_to_function(func, init_DoE)
    return mlcs.BiFidelityDoE(*DoE)


def get_experiment_subsampled_EG(func, DoE, instances):

    results = []
    for num_high, num_low, rep in instances:
        mlcs.set_seed_by_instance(num_high, num_low, rep)

        # Create sub-sampled Multi-Fidelity DoE in- and output according to instance specification
        train, test = mlcs.split_bi_fidelity_doe(DoE, num_high, num_low)
        train_high_y, train_low_y = func.high(train.high), \
                                    func.low(train.low)

        # Create an archive from the MF-function and MF-DoE data
        archive = mlcs.CandidateArchive.from_multi_fidelity_function(func)
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

    eg = mlcs.ProtoEG(archive, num_reps=num_reps, interval=1)
    eg.subsample_errorgrid()
    return eg


@composite
def valid_subsample_spec(draw):
    max_high = draw(integers(min_value=2, max_value=1_000))
    max_low = draw(integers(min_value=max_high, max_value=10_000))
    num_high = draw(integers(min_value=1, max_value=max_high-1))
    num_low = draw(integers(min_value=num_high, max_value=max_low))

    return num_high, num_low, max_high, max_low


@given(valid_subsample_spec())
def test_calc_reuse_fraction_high(spec):
    num_high, num_low, max_high, max_low = spec
    peg = mlcs.ProtoEG(archive=mlcs.CandidateArchive())

    part1 = binom(max_high, num_high)
    part2 = binom(max_high + 1, num_high)

    if not (np.isfinite(part1) and np.isfinite(part2)):
        return  # invalid input that cannot be tested

    true_fraction = part1 / part2
    fraction = peg.calculate_reuse_fraction(num_high, num_low, fidelity='high',
                                            max_high=max_high, max_low=max_low)

    assert fraction == approx(true_fraction)


@given(valid_subsample_spec())
def test_calc_reuse_fraction_low(spec):
    num_high, num_low, max_high, max_low = spec
    peg = mlcs.ProtoEG(archive=mlcs.CandidateArchive())

    part1 = binom(max_low - num_high, num_low - num_high)
    part2 = binom(max_low+1 - num_high, num_low - num_high)

    if not (np.isfinite(part1) and np.isfinite(part2)):
        return  # invalid input that cannot be tested

    true_fraction = part1 / part2
    fraction = peg.calculate_reuse_fraction(num_high, num_low, fidelity='low',
                                            max_high=max_high, max_low=max_low)

    assert fraction == approx(true_fraction)


def test_experiment():

    func = mf2.currin
    num_reps = 1
    DoE = prepare_DoE(func)

    spec = mlcs.InstanceSpec(len(DoE[0])-1, len(DoE[1]), num_reps=num_reps)
    instances = list(spec.instances)
    eg2 = get_experiment_subsampled_EG(func, DoE, instances=instances)

    np.testing.assert_allclose(
        [0.27112833049506163],
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
        0.7159078759175921,
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
    archive.addcandidate(candidate=new_sample.flatten(), fitness=func.low(new_sample), fidelity='low')

    prev_coords = proto_eg.error_grid.coords
    proto_eg.update_errorgrid_with_sample(new_sample, fidelity='low')
    assert len(proto_eg.error_grid.coords['n_low'].values) > len(prev_coords['n_low'].values)


def test_protoEG_subsample_errorgrid_update_high():

    func = mf2.currin
    num_reps = 1
    DoE_high, DoE_low = prepare_DoE(func)

    archive = mlcs.CandidateArchive.from_multi_fidelity_function(func)
    archive.addcandidates(DoE_high, func.high(DoE_high), fidelity='high')
    archive.addcandidates(DoE_low, func.low(DoE_low), fidelity='low')
    proto_eg = get_subsampled_protoEG(archive, num_reps)

    np.random.seed(0)
    non_high = set(tuple(c) for c in DoE_low) - set(tuple(c) for c in DoE_high)
    new_sample = np.array(next(iter(non_high))).reshape(1, -1) # just take 1 element
    archive.addcandidate(candidate=new_sample.flatten(), fitness=func.high(new_sample), fidelity='high')

    prev_coords = proto_eg.error_grid.coords
    proto_eg.update_errorgrid_with_sample(new_sample, fidelity='high')
    assert len(proto_eg.error_grid.coords['n_high'].values) > len(prev_coords['n_high'].values)
