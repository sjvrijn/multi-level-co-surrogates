#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
test_CandidateArchive.py: Set of tests for the mlcs.CandidateArchive
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'

import pytest
import multiLevelCoSurrogates as mlcs
import numpy as np
from multiLevelCoSurrogates import CandidateArchive


def test_bare_archive():
    archive = CandidateArchive(ndim=0)
    assert archive.fidelities == ('fitness',)
    assert len(archive) == len(archive.data) == 0
    assert len(archive.max) == len(archive.min) == len(archive.fidelities)


def test_single_fidelity():
    fid = 'my_fidelity'
    archive = CandidateArchive(ndim=0, fidelities=(fid,))
    assert archive.fidelities == (fid,)


def test_multiple_fidelities():
    fids = [f'my_{i}th_fidelity' for i in range(5)]
    archive = CandidateArchive(ndim=0, fidelities=fids)
    assert archive.fidelities == fids


def test_fidelity_not_specified():
    fids = [f'my_{i}th_fidelity' for i in range(5)]
    archive = CandidateArchive(ndim=0, fidelities=fids)
    with pytest.raises(ValueError):
        archive.add_candidate([1, 2, 3], fitness=1)


### A 'happy path' is a simple run through some functionality that just works

def test_1fid_happy_path():
    archive = CandidateArchive(ndim=3)
    candidates = np.random.randn(30).reshape((10, 3))
    fitnesses = np.random.randn(10).reshape((10, 1))
    archive.add_candidates(candidates.tolist(), fitnesses)

    result = archive.get_candidates()
    assert isinstance(result, mlcs.CandidateSet)
    assert hasattr(result, 'candidates')
    assert hasattr(result, 'fitnesses')

    cand, fit = result
    np.testing.assert_array_almost_equal(candidates, cand)
    np.testing.assert_array_almost_equal(fitnesses, fit)


def test_2fid_happy_path():
    archive = CandidateArchive(ndim=3, fidelities=('AAA', 'BBB'))
    candidates = np.random.randn(30).reshape((10, 3))
    fitnesses = np.random.randn(10).reshape((10, 1))
    with pytest.raises(ValueError):
        archive.add_candidates(candidates, fitnesses)

    archive.add_candidates(candidates.tolist(), fitnesses, fidelity='AAA')

    cand, fit = archive.get_candidates(fidelity='AAA')
    np.testing.assert_array_almost_equal(candidates, cand)
    np.testing.assert_array_almost_equal(fitnesses, fit)

    new_fitnesses = np.random.randn(5).reshape((5, 1))
    indices = np.random.choice(np.arange(10), 5, replace=False)

    archive.add_candidates(candidates[indices].tolist(), new_fitnesses, fidelity='BBB')

    cand, fit = archive.get_candidates(fidelity='BBB')
    # comparing sorted because order does not matter...
    np.testing.assert_array_almost_equal(
        sorted(candidates[indices].tolist()),
        sorted(cand.tolist())
    )
    np.testing.assert_array_almost_equal(sorted(new_fitnesses), sorted(fit))
