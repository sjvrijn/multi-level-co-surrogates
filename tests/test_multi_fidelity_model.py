#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
test_multi_fidelity_model.py: Set of tests for the mlcs.surrogate.MultiFidelityModel
"""

__author__ = 'Sander van Rijn'
__email__ = 's.j.van.rijn@liacs.leidenuniv.nl'


import numpy as np
import mf2
import multiLevelCoSurrogates as mlcs


def test_simple_path():
    """Quick test case recreating the 1D forrester model"""

    np.random.seed(20160501)  # unbiased random seed

    low_x = np.linspace(0,1,11).reshape(-1,1)
    high_x = low_x[[0,4,6,10]]
    archive = mlcs.CandidateArchive(['high', 'low', 'high-low'])
    archive.addcandidates(low_x, mf2.forrester.low(low_x), fidelity='low')
    archive.addcandidates(high_x, mf2.forrester.high(high_x), fidelity='high')
    mfm = mlcs.MultiFidelityModel(mf2.forrester.fidelity_names, archive)

    test_x = high_x + 0.05
    test_y = mfm.top_level_model.predict(test_x)

    expected = [1.2319664471068705, 1.9409778220340197, -4.0374255643707695, 18.010376232109593]

    np.testing.assert_almost_equal(test_y, expected, decimal=5)

