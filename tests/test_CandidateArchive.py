#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
test_CandidateArchive.py: Set of tests for the mlcs.CandidateArchive
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

from collections import namedtuple

from hypothesis import given
from hypothesis.strategies import lists, text, integers
import numpy as np
import pytest
from more_itertools import pairwise

import multiLevelCoSurrogates as mlcs
from multiLevelCoSurrogates import CandidateArchive, CandidateArchiveNew


implementations = [
    CandidateArchive,
    CandidateArchiveNew,
]


def setup_archive(Archive):
    archive = Archive(fidelities=['A', 'B', 'C'])
    all_candidates = np.random.rand(10, 2)
    data = {
        'A': (all_candidates[:3], np.random.rand(3)),
        'B': (all_candidates, np.random.rand(10)),
        'C': (all_candidates[3:], np.random.rand(7)),
    }
    for fidelity, (candidates, fitness) in data.items():
        archive.addcandidates(candidates, fitness, fidelity=fidelity)
    return all_candidates, archive, data


@pytest.mark.parametrize('Archive', implementations)
def test_bare_archive(Archive):
    archive = Archive()
    assert len(archive) == 0


@pytest.mark.parametrize('Archive', implementations)
def test_add_candidate_increases_length(Archive):
    archive = Archive()
    old_length = len(archive)

    archive.addcandidate(np.random.rand(2), np.random.random())
    assert len(archive) == old_length + 1


@pytest.mark.parametrize('Archive', implementations)
def test_add_candidates_increases_length(Archive):
    archive1 = Archive()
    archive2 = Archive()
    num_candidates = 5

    for _ in range(num_candidates):
        archive1.addcandidate(np.random.rand(2), np.random.random())

    archive2.addcandidates(np.random.rand(5, 2), np.random.rand(5))

    assert len(archive1) == len(archive2) == num_candidates


@pytest.mark.parametrize('Archive', implementations)
def test_add_same_candidate_maintains_length(Archive):
    archive = Archive()
    candidate, fitness = np.random.rand(2), np.random.random()
    archive.addcandidate(candidate, fitness)
    old_length = len(archive)

    # adding exact same candidate does not change length
    archive.addcandidate(candidate, fitness)
    assert len(archive) == old_length

    # adding same candidate with different fitness does not change length
    archive.addcandidate(candidate, np.random.random())
    assert len(archive) == old_length


@pytest.mark.parametrize('Archive', implementations)
def test_add_candidate_without_fidelity_raises_error(Archive):
    all_candidates, archive, data = setup_archive(Archive)

    candidates = np.random.rand(5, 2)
    fitnesses = np.random.rand(5, 1)

    with pytest.raises(ValueError):
        archive.addcandidate(candidates[0], fitnesses[0])
    with pytest.raises(ValueError):
        archive.addcandidates(candidates, fitnesses)


@pytest.mark.parametrize('Archive', implementations)
def test_from_bifiddoe(Archive):
    ndim, num_high, num_low = 2, 5, 10
    doe = mlcs.bi_fidelity_doe(ndim, num_high, num_low)

    archive = Archive.from_bi_fid_DoE(*doe, np.random.rand(num_high), np.random.rand(num_low))
    assert len(archive) == num_low
    assert archive.count('high') == num_high
    assert archive.count('low') == num_low


@pytest.mark.parametrize('Archive', implementations)
def test_getfitnesses_one_fid(Archive):
    all_candidates, archive, data = setup_archive(Archive)

    # retrieve exactly the fitnesses that were input for these candidates
    for fidelity, (candidates, fitness) in data.items():
        stored_fitness = archive.getfitnesses(candidates, fidelity=fidelity)
        assert np.allclose(stored_fitness, fitness)

    # retrieve this fitness for all candidates, assert #input are not NaN
    for fidelity, (_, fitness) in data.items():
        stored_fitness = archive.getfitnesses(all_candidates, fidelity=fidelity)
        assert np.count_nonzero(~np.isnan(stored_fitness)) == len(fitness)


@pytest.mark.parametrize('Archive', implementations)
def test_getfitnesses_multiple_fid(Archive):
    all_candidates, archive, data = setup_archive(Archive)

    for fidelities in pairwise(data.keys()):
        stored_fitness = archive.getfitnesses(all_candidates, fidelity=fidelities)

        assert stored_fitness.shape == (len(all_candidates), len(fidelities))

        # check that fidelity order as columns is correct
        for i, fidelity in enumerate(fidelities):
            column = stored_fitness[:, i]
            assert np.count_nonzero(~np.isnan(column)) == len(data[fidelity][1])


@pytest.mark.parametrize('Archive', implementations)
def test_1fid_getcandidates(Archive):
    archive = Archive()
    candidates = np.random.rand(10, 3)
    fitnesses = np.random.rand(10, 1)
    archive.addcandidates(candidates.tolist(), fitnesses)

    result = archive.getcandidates()
    assert hasattr(result, 'candidates')
    assert hasattr(result, 'fitnesses')

    cand, fit = result
    np.testing.assert_array_almost_equal(candidates, cand)
    np.testing.assert_array_almost_equal(fitnesses, fit)


@pytest.mark.parametrize('Archive', implementations)
def test_2fid_getcandidates(Archive):
    ndim = 3
    archive = Archive(fidelities=['AAA', 'BBB'])

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
