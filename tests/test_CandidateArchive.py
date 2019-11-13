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
from multiLevelCoSurrogates import CandidateArchive


def test_bare_archive():
    archive = CandidateArchive(ndim=0)
    assert archive.fidelities == ['fitness']
    assert len(archive) == len(archive.data) == 0
    assert len(archive.max) == len(archive.min) == len(archive.fidelities)


def test_single_fidelity():
    fid = 'my_fidelity'
    archive = CandidateArchive(ndim=0, fidelities=[fid])
    assert archive.fidelities == [fid]


def test_multiple_fidelities():
    fids = [f'my_{i}th_fidelity' for i in range(5)]
    archive = CandidateArchive(ndim=0, fidelities=fids)
    assert archive.fidelities == fids


def test_fidelity_not_specified():
    fids = [f'my_{i}th_fidelity' for i in range(5)]
    archive = CandidateArchive(ndim=0, fidelities=fids)
    with pytest.raises(ValueError):
        archive.addcandidate([1, 2, 3], fitness=1)


MultiFidFunc = namedtuple('MultiFidFunc', 'ndim fidelity_names')

@given(lists(text(), min_size=2), integers())
def test_from_mff(fidelities, ndim):
    mff = MultiFidFunc(ndim, fidelities)

    archive = CandidateArchive.from_multi_fidelity_function(mff)

    # Each of the n-1 consecutive pairs has to be added as a new fidelity,
    # so len(archive.fidelities) should be n + (n-1) = 2n - 1
    assert len(archive.fidelities) == 2*len(fidelities) - 1
    assert archive.ndim == ndim

    mff = MultiFidFunc(0.5, fidelities)
    archive = CandidateArchive.from_multi_fidelity_function(mff, ndim=ndim)
    # archive.ndim should not be the non-integer value of 0.5 when overwritten
    assert archive.ndim == ndim


### A 'happy path' is a simple run through some functionality that just works

def test_1fid_happy_path():
    archive = CandidateArchive(ndim=3)
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


def test_2fid_happy_path():
    archive = CandidateArchive(ndim=3, fidelities=['AAA', 'BBB'])
    candidates = np.random.randn(30).reshape((10, 3))
    fitnesses = np.random.randn(10).reshape((10, 1))
    with pytest.raises(ValueError):
        archive.addcandidates(candidates, fitnesses)

    archive.addcandidates(candidates.tolist(), fitnesses, fidelity='AAA')

    cand, fit = archive.getcandidates(fidelity='AAA')
    np.testing.assert_array_almost_equal(candidates, cand)
    np.testing.assert_array_almost_equal(fitnesses, fit)

    new_fitnesses = np.random.randn(5).reshape((5, 1))
    indices = np.random.choice(np.arange(10), 5, replace=False)

    archive.addcandidates(candidates[indices].tolist(), new_fitnesses, fidelity='BBB')

    cand, fit = archive.getcandidates(fidelity='BBB')
    # comparing sorted because order does not matter...
    np.testing.assert_array_almost_equal(
        sorted(candidates[indices].tolist()),
        sorted(cand.tolist())
    )
    np.testing.assert_array_almost_equal(sorted(new_fitnesses), sorted(fit))
