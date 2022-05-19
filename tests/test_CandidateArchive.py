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
]


@pytest.mark.parametrize('Archive', implementations)
def test_bare_archive(Archive):
    archive = Archive()
    assert len(archive) == 0


### A 'happy path' is a simple run through some functionality that just works

@pytest.mark.parametrize('Archive', implementations)
def test_1fid_happy_path(Archive):
    archive = Archive()
    candidates = np.random.randn(30).reshape((10, 3))
    fitnesses = np.random.randn(10).reshape((10, 1))
    archive.addcandidates(candidates.tolist(), fitnesses)

    result = archive.getcandidates()
    assert isinstance(result, mlcs.CandidateSet)
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
