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

import experiments as exp
import multiLevelCoSurrogates as mlcs



def prepare_DoE(func, nh=3, nl=4):
    np.random.seed(20160501)  # Setting seed for reproducibility
    init_DoE = mlcs.bi_fidelity_doe(func.ndim, nh, nl)
    DoE = exp.scale_to_function(func, init_DoE)
    return DoE


def get_experiment_subsampled_EG(func, DoE, instances):

    results = []
    for i, (num_high, num_low, rep) in enumerate(instances):

        mlcs.set_seed_by_instance(num_high, num_low, rep)

        # Create sub-sampled Multi-Fidelity DoE in- and output according to instance specification
        train, test = mlcs.split_bi_fidelity_doe(DoE, num_high, num_low)
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


@composite
def valid_subsample_spec(draw):
    max_high = draw(integers(min_value=2, max_value=1_000))
    max_low = draw(integers(min_value=max_high, max_value=10_000))
    num_high = draw(integers(min_value=1, max_value=max_high-1))
    num_low = draw(integers(min_value=num_high, max_value=max_low))

    return num_high, num_low, max_high, max_low


def test_split_doe_with_include_high():
    doe = mlcs.bi_fidelity_doe(2, 10, 20)
    X = np.random.rand(1, 2)
    num_high, num_low = 5, 10
    selected, other = mlcs.split_with_include(doe, num_high, num_low, X, 'high')

    assert len(selected.low) == num_low
    assert len(selected.high) == num_high
    assert len(other.low) == len(doe.low) - num_low + 1
    assert len(other.high) == len(doe.high) - num_high + 1

    X = tuple(X[0])
    assert X in set(tuple(x) for x in selected.low)
    assert X in set(tuple(x) for x in selected.high)
    assert X not in set(tuple(x) for x in other.low)
    assert X not in set(tuple(x) for x in other.high)



def test_split_doe_with_include_low():
    doe = mlcs.bi_fidelity_doe(2, 10, 20)
    X = np.random.rand(1, 2)
    num_high, num_low = 5, 10
    selected, other = mlcs.split_with_include(doe, num_high, num_low, X, 'low')

    assert len(selected.low) == num_low
    assert len(selected.high) == num_high
    assert len(other.low) == len(doe.low) - num_low + 1
    assert len(other.high) == len(doe.high) - num_high

    X = tuple(X[0])
    assert X in set(tuple(x) for x in selected.low)
    assert X not in set(tuple(x) for x in selected.high)
    assert X not in set(tuple(x) for x in other.low)
    assert X not in set(tuple(x) for x in other.high)


@given(valid_subsample_spec())
def test_calc_reuse_fraction_high(spec):
    num_high, num_low, max_high, max_low = spec
    peg = mlcs.ProtoEG(archive=mlcs.CandidateArchive(ndim=0))

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
    peg = mlcs.ProtoEG(archive=mlcs.CandidateArchive(ndim=0))

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
    new_sample = np.array(next(iter(non_high))).reshape(1, -1) # just take 1 element
    proto_eg.update_errorgrid_with_sample(new_sample, func.high(new_sample), fidelity='high')
    assert archive.count('high') == len(DoE_high) + 1
