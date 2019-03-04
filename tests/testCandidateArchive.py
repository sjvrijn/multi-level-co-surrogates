#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
testCandidateArchive.py: Set of tests for the mlcs.CandidateArchive
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


import multiLevelCoSurrogates as mlcs
from multiLevelCoSurrogates import CandidateArchive


def test_bare_archive():
    archive = CandidateArchive(ndim=0)
    assert archive.fidelities == ['fitness']


def test_single_fidelity():
    fid = 'my_fidelity'
    archive = CandidateArchive(ndim=0, fidelities=[fid])
    assert archive.fidelities == [fid]


def test_multiple_fidelities():
    fids = [f'my_{i}th_fidelity' for i in range(5)]
    archive = CandidateArchive(ndim=0, fidelities=fids)
    assert archive.fidelities == fids



