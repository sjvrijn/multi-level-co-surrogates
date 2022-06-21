#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
test_CandidateArchive.py: Set of tests for the mlcs.CandidateArchive
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


from hypothesis import given
from hypothesis.strategies import lists, text, integers
import numpy as np
from pytest import fixture, raises, warns
from more_itertools import pairwise

import multiLevelCoSurrogates as mlcs
from multiLevelCoSurrogates import CandidateArchive


def setup_archive():
    archive = CandidateArchive(fidelities=['A', 'B', 'C'])
    all_candidates = np.random.rand(10, 2)
    data = {
        'A': (all_candidates[:3], np.random.rand(3)),
        'B': (all_candidates, np.random.rand(10)),
        'C': (all_candidates[3:], np.random.rand(7)),
    }
    for fidelity, (candidates, fitness) in data.items():
        archive.addcandidates(candidates, fitness, fidelity=fidelity)
    return all_candidates, archive, data


@fixture
def bifid_archive():
    """Create a typical bi-fidelity archive with fidelities 'high' and 'low'"""
    archive = CandidateArchive()
    candidates = np.random.rand(10, 2)
    archive.addcandidates(candidates, np.random.rand(10), fidelity='low')
    archive.addcandidates(candidates[:5], np.random.rand(5), fidelity='high')

    return archive


def test_bare_archive():
    archive = CandidateArchive()
    assert len(archive) == 0


def test_add_candidate_increases_length():
    archive = CandidateArchive()
    old_length = len(archive)

    archive.addcandidate(np.random.rand(2), np.random.random())
    assert len(archive) == old_length + 1


def test_add_candidates_increases_length():
    archive1 = CandidateArchive()
    archive2 = CandidateArchive()
    num_candidates = 5

    for _ in range(num_candidates):
        archive1.addcandidate(np.random.rand(2), np.random.random())

    archive2.addcandidates(np.random.rand(5, 2), np.random.rand(5))

    assert len(archive1) == len(archive2) == num_candidates


def test_add_same_candidate_maintains_length():
    archive = CandidateArchive()
    candidate, fitness = np.random.rand(2), np.random.random()
    archive.addcandidate(candidate, fitness)
    old_length = len(archive)

    # adding exact same candidate does not change length
    archive.addcandidate(candidate, fitness)
    assert len(archive) == old_length

    # adding same candidate with different fitness does not change length
    archive.addcandidate(candidate, np.random.random())
    assert len(archive) == old_length


def test_min_max():
    all_candidates, archive, data = setup_archive()

    for fidelity, (_, fitnesses) in data.items():
        assert archive.max[fidelity] == np.max(fitnesses)
        assert archive.min[fidelity] == np.min(fitnesses)


def test_add_candidate_without_fidelity_raises_error():
    """Adding candidates of unspecified fidelity to an archive with previously
     specified fidelities should raise ValueError.
     """
    _, archive, _ = setup_archive()

    candidates = np.random.rand(5, 2)
    fitnesses = np.random.rand(5, 1)

    with raises(ValueError):
        archive.addcandidate(candidates[0], fitnesses[0])
    with raises(ValueError):
        archive.addcandidates(candidates, fitnesses)


def test_add_candidate_with_nan_raises_error():
    """Adding candidates with `NaN` as fitness raises ValueError"""
    archive = CandidateArchive()
    with raises(ValueError):
        archive.addcandidate(np.random.rand(5), np.nan)

    fitnesses = np.random.rand(5)
    fitnesses[2] = np.nan
    with raises(ValueError):
        archive.addcandidates(np.random.rand(5,5), fitnesses)


def test_from_bifiddoe():
    ndim, num_high, num_low = 2, 5, 10
    doe = mlcs.bi_fidelity_doe(ndim, num_high, num_low)

    archive = CandidateArchive.from_bi_fid_doe(*doe, np.random.rand(num_high), np.random.rand(num_low))
    assert len(archive) == num_low
    assert archive.count('high') == num_high
    assert archive.count('low') == num_low


def test_as_doe():
    archive = CandidateArchive(fidelities=['high', 'low'])

    high_x = np.random.rand(10, 2)
    archive.addcandidates(high_x, np.random.rand(10), fidelity='high')

    low_x = np.random.rand(20, 2)
    archive.addcandidates(low_x, np.random.rand(20), fidelity='low')

    doe = archive.as_doe()

    assert hasattr(doe, 'high')
    assert hasattr(doe, 'low')

    np.testing.assert_array_almost_equal(doe.high, high_x)
    np.testing.assert_array_almost_equal(doe.low, low_x)


def test_getfitnesses_one_fid():
    all_candidates, archive, data = setup_archive()

    # retrieve exactly the fitnesses that were input for these candidates
    for fidelity, (candidates, fitness) in data.items():
        stored_fitness = archive.getfitnesses(candidates, fidelity=fidelity)
        assert np.allclose(stored_fitness, fitness)

    # retrieve this fitness for all candidates, assert #input are not NaN
    for fidelity, (_, fitness) in data.items():
        stored_fitness = archive.getfitnesses(all_candidates, fidelity=fidelity)
        assert np.count_nonzero(~np.isnan(stored_fitness)) == len(fitness)


def test_getfitnesses_multiple_fid():
    all_candidates, archive, data = setup_archive()

    for fidelities in pairwise(data.keys()):
        stored_fitness = archive.getfitnesses(all_candidates, fidelity=fidelities)

        assert stored_fitness.shape == (len(all_candidates), len(fidelities))

        # check that fidelity order as columns is correct
        for i, fidelity in enumerate(fidelities):
            column = stored_fitness[:, i]
            assert np.count_nonzero(~np.isnan(column)) == len(data[fidelity][1])


def test_1fid_getcandidates():
    archive = CandidateArchive()
    candidates = np.random.rand(10, 3)
    fitnesses = np.random.rand(10, 1)
    archive.addcandidates(candidates.tolist(), fitnesses)

    result = archive.getcandidates()
    assert hasattr(result, 'candidates')
    assert hasattr(result, 'fitnesses')

    cand, fit = result
    np.testing.assert_array_almost_equal(candidates, cand)
    np.testing.assert_array_almost_equal(fitnesses, fit)


def test_2fid_getcandidates():
    ndim = 3
    archive = CandidateArchive(fidelities=['AAA', 'BBB'])

    num_candidates = 10
    candidates = np.random.rand(num_candidates, ndim)
    fitnesses = np.random.rand(num_candidates, 1)
    archive.addcandidates(candidates.tolist(), fitnesses, fidelity='AAA')

    cand, fit = archive.getcandidates(fidelity='AAA')
    np.testing.assert_array_almost_equal(candidates, cand)
    np.testing.assert_array_almost_equal(fitnesses, fit)

    num_fitnesses_BBB = 5
    new_fitnesses = np.random.rand(num_fitnesses_BBB, 1)
    indices = np.random.choice(np.arange(10), num_fitnesses_BBB, replace=False)

    archive.addcandidates(candidates[indices].tolist(), new_fitnesses, fidelity='BBB')

    cand, fit = archive.getcandidates(fidelity='BBB')
    # comparing sorted because order does not matter...
    np.testing.assert_array_almost_equal(
        sorted(candidates[indices].tolist()),
        sorted(cand.tolist())
    )
    np.testing.assert_array_almost_equal(sorted(new_fitnesses), sorted(fit))


def test_2fid_getcandidates_fidelity_list():
    """When passing multiple fidelities, results should be for candidates
    for which *ALL* fidelities are known/present"""
    ndim = 3
    archive = CandidateArchive(fidelities=['AAA', 'BBB'])

    num_candidates = 10
    candidates = np.random.rand(num_candidates, ndim)
    fitnesses = np.random.rand(num_candidates, 1)
    archive.addcandidates(candidates.tolist(), fitnesses, fidelity='AAA')

    num_fitnesses_BBB = 5
    new_fitnesses = np.random.rand(num_fitnesses_BBB, 1)
    indices = np.random.choice(np.arange(10), num_fitnesses_BBB, replace=False)
    archive.addcandidates(candidates[indices].tolist(), new_fitnesses, fidelity='BBB')

    cand, fit = archive.getcandidates(['AAA', 'BBB'])
    assert len(fit) == num_fitnesses_BBB
    assert np.count_nonzero(np.isnan(fit)) == 0


def test_save(tmp_path):
    all_candidates, archive, data = setup_archive()

    test_path = tmp_path / 'test_archive.npz'
    archive.save(test_path)

    loaded_data = np.load(test_path)

    assert np.allclose(all_candidates, loaded_data['candidates'])
    assert all(fid in loaded_data for fid in archive.fidelities)


def test_roundtrip(tmp_path):
    all_candidates, archive, data = setup_archive()

    test_path = tmp_path / 'test_archive.npz'
    archive.save(test_path)

    new_archive = CandidateArchive.from_file(test_path)

    assert len(archive) == len(new_archive)
    assert archive.min == new_archive.min
    assert archive.max == new_archive.max
    assert list(archive.fidelities) == list(new_archive.fidelities)

    for fid in archive.fidelities:
        assert archive.count(fid) == new_archive.count(fid)


def test_undo_last():
    archive = CandidateArchive()
    archive.addcandidate(np.random.rand(2), np.random.rand(), fidelity='A')
    archive.addcandidate(np.random.rand(2), np.random.rand(), fidelity='B')
    archive.addcandidate(np.random.rand(2), np.random.rand(), fidelity='A')

    archive.undo_last()
    assert archive.count('A') == 1
    assert archive.count('B') == 1


def test_undo_last_overwritten():
    archive = CandidateArchive()
    overwrite_candidate = np.random.rand(2)
    archive.addcandidate(np.random.rand(2), np.random.rand(), fidelity='A')
    archive.addcandidate(np.random.rand(2), np.random.rand(), fidelity='B')
    archive.addcandidate(overwrite_candidate, np.random.rand(), fidelity='A')
    archive.addcandidate(overwrite_candidate, np.random.rand(), fidelity='A')

    archive.undo_last()
    assert archive.count('A') == 1
    assert archive.count('B') == 1


def test_undo_last_specify_fidelity():
    archive = CandidateArchive()
    fidelity_order = 'AABBBBBAA'
    for fid in fidelity_order:
        archive.addcandidate(np.random.rand(2), np.random.rand(), fidelity=fid)

    archive.undo_last(fidelity='B')
    assert archive.count('A') == 2
    assert archive.count('B') == 4

    archive.undo_last(fidelity='A')
    assert archive.count('A') == 1
    assert archive.count('B') == 0


def test_undo_last_incorrect_fidelity():

    archive = CandidateArchive()
    fidelity_order = 'AABB'
    for fid in fidelity_order:
        archive.addcandidate(np.random.rand(2), np.random.rand(), fidelity=fid)

    with raises(ValueError):
        archive.undo_last(fidelity='C')


def test_split_archive(bifid_archive):
    a_high, a_low = 3, 7
    a, b = bifid_archive.split(3, 7)

    assert a.count('low') == a_low
    assert a.count('high') == a_high
    assert a.count('low') + b.count('low') == bifid_archive.count('low')
    assert a.count('high') + b.count('high') == bifid_archive.count('high')


def test_split_errors(bifid_archive):
    # invalid num_high
    with raises(ValueError):
        bifid_archive.split(-1, 7)
    with raises(ValueError):
        bifid_archive.split(6, 7)

    # invalid num_low
    with raises(ValueError):
        bifid_archive.split(3, 11)

    # # invalid comparison
    with raises(ValueError):
        bifid_archive.split(4, 3)


def test_split_warnings(bifid_archive):

    # not enough high-fid
    with warns(mlcs.LowHighFidSamplesWarning):
        bifid_archive.split(0, 3)
    with warns(mlcs.LowHighFidSamplesWarning):
        bifid_archive.split(1, 3)

    # no high-fid samples in test-set
    with warns(mlcs.NoHighFidTrainSamplesWarning):
        bifid_archive.split(5, 7)

    # no non-high-fidelity low-fidelity samples in test set
    with warns(mlcs.NoSpareLowFidSamplesWarning):
        bifid_archive.split(3, 3)
