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

import multiLevelCoSurrogates as mlcs
from multiLevelCoSurrogates import CandidateArchive, CandidateArchiveNew


implementations = [
    CandidateArchive,
    CandidateArchiveNew,
]


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
def test_from_bifiddoe(Archive):
    ndim, num_high, num_low = 2, 5, 10
    doe = mlcs.bi_fidelity_doe(ndim, num_high, num_low)

    archive = Archive.from_bi_fid_DoE(*doe, np.random.rand(num_high), np.random.rand(num_low))
    assert len(archive) == num_low


### A 'happy path' is a simple run through some functionality that just works

@pytest.mark.parametrize('Archive', implementations)
def test_1fid_happy_path(Archive):
    archive = Archive()
    candidates = np.random.randn(30).reshape((10, 3))
    fitnesses = np.random.randn(10).reshape((10, 1))
    archive.addcandidates(candidates.tolist(), fitnesses)

    result = archive.getcandidates()
    assert hasattr(result, 'candidates')
    assert hasattr(result, 'fitnesses')

    cand, fit = result
    np.testing.assert_array_almost_equal(candidates, cand)
    np.testing.assert_array_almost_equal(fitnesses, fit)


@pytest.mark.parametrize('Archive', implementations)
def test_2fid_happy_path(Archive):
    ndim = 3
    archive = Archive(fidelities=['AAA', 'BBB'])

    num_candidates = 10
    candidates = np.random.randn(num_candidates*ndim).reshape((num_candidates, ndim))
    fitnesses = np.random.randn(num_candidates).reshape((num_candidates, 1))
    with pytest.raises(ValueError):
        archive.addcandidates(candidates, fitnesses)

    archive.addcandidates(candidates.tolist(), fitnesses, fidelity='AAA')
    assert archive.count('AAA') == num_candidates
    assert archive.count('BBB') == 0

    cand, fit = archive.getcandidates(fidelity='AAA')
    np.testing.assert_array_almost_equal(candidates, cand)
    np.testing.assert_array_almost_equal(fitnesses, fit)

    num_fitnesses_BBB = 5
    new_fitnesses = np.random.randn(num_fitnesses_BBB).reshape((num_fitnesses_BBB, 1))
    indices = np.random.choice(np.arange(10), num_fitnesses_BBB, replace=False)

    archive.addcandidates(candidates[indices].tolist(), new_fitnesses, fidelity='BBB')
    assert archive.count('AAA') == num_candidates
    assert archive.count('BBB') == num_fitnesses_BBB

    cand, fit = archive.getcandidates(fidelity='BBB')
    # comparing sorted because order does not matter...
    np.testing.assert_array_almost_equal(
        sorted(candidates[indices].tolist()),
        sorted(cand.tolist())
    )
    np.testing.assert_array_almost_equal(sorted(new_fitnesses), sorted(fit))
